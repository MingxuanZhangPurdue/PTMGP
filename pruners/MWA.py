import torch
import math
import re
import warnings
from composer.core import Algorithm, Event
from pruners.utils_composer import _convert_timestr_to_int

def _count_mask_differences(mask1, mask2):
    n_same = 0
    n_total = 0
    for key in mask1.keys():
        n_total += mask1[key].numel()
        n_same += torch.eq(mask1[key], mask2[key]).sum().item()
    n_diff = n_total - n_same
    return n_diff

class MWA(Algorithm):
    def __init__(
            self,
            train_size,
            max_train_steps,
            sigma0=1e-10,
            sigma1=0.05,
            lambda_mix=1e-1,
            alpha_i_lambda_mix=1.0,
            alpha_f_lambda_mix=0.001,
            initial_sparsity=0.0,
            final_sparsity=0.0,
            initial_warmup_steps=0,
            pruning_interval=1000,
            pruning_params=None,
            sparse_finetune_steps=0,
            log_interval=None,
        ):

        assert initial_sparsity < 1.0 and initial_sparsity >= 0.0, "initial_sparsity must be in the range [0, 1)"
        assert final_sparsity < 1.0 and final_sparsity >= 0.0, "final_sparsity must be in the range [0, 1)"
        assert initial_sparsity <= final_sparsity, "initial_sparsity must be less than or equal to final_sparsity"

        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity

        pruning_start = initial_warmup_steps
        pruning_end = max_train_steps - sparse_finetune_steps - 1

        assert 0 <= pruning_start < pruning_end, (
            f"pruning_start: {pruning_start}, "
            f"pruning_end: {pruning_end}, "
            "Condition 0 <= pruning_start < pruning_end must be satisfied, but got False"
        )

        self.pruning_start = pruning_start
        self.pruning_end = pruning_end
        self.pruning_interval = pruning_interval

        self.n_total_param_for_pruning = 0
        self.after_initial_warmup_mask = None
        self.current_mask = None
        self.final_fixed_mask = None
        self.current_prior_threshold = None
        self.current_sparsity_mask = None

        self.train_size = train_size

        self.pruning_params = re.compile("|".join(pruning_params), re.IGNORECASE) if pruning_params is not None else None

        if log_interval is not None and log_interval % pruning_interval != 0:
            warnings.warn (
                f"log_interval: {log_interval}, "
                f"pruning_interval: {pruning_interval}. "
                "When log_interval is not None, log_interval must be divisible by pruning_interval, "
                "otherwise, only some of the research-related metrics will be logged. "
                "For more details, please refer to the apply function."
            )
        
        self.log_interval = log_interval

        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.lambda_mix = lambda_mix
        self.alpha_i_lambda_mix = alpha_i_lambda_mix
        self.alpha_f_lambda_mix = alpha_f_lambda_mix

    # initialize the algorithm from the command line arguments
    @classmethod
    def from_args(self, train_size, max_train_steps, train_dataloader_len, args):
        initial_warmup_steps = _convert_timestr_to_int(args.initial_warmup_steps, max_train_steps, train_dataloader_len)
        sparse_finetune_steps = _convert_timestr_to_int(args.sparse_finetune_steps, max_train_steps, train_dataloader_len)
        pruning_interval = _convert_timestr_to_int(args.pruning_interval, max_train_steps, train_dataloader_len)
        log_interval = _convert_timestr_to_int(args.log_interval, max_train_steps, train_dataloader_len) if args.log_interval is not None else None
        return self(
            train_size=train_size,
            max_train_steps=max_train_steps,
            sigma0=args.sigma0,
            sigma1=args.sigma1,
            lambda_mix=args.lambda_mix,
            alpha_i_lambda_mix=args.alpha_i_lambda_mix,
            alpha_f_lambda_mix=args.alpha_f_lambda_mix,
            initial_sparsity=args.initial_sparsity,
            final_sparsity=args.final_sparsity,
            initial_warmup_steps=initial_warmup_steps,
            pruning_interval=pruning_interval,
            pruning_params=args.pruning_params,
            sparse_finetune_steps=sparse_finetune_steps,
            log_interval=log_interval
        )

    def whether_prune_param(self, n):
        if self.pruning_params == None:
            return True
        else:
            return bool(re.search(self.pruning_params, n))
        
    def annealing_scheduler(self, train_step_index):
        if train_step_index < self.pruning_start:
            alpha = 1.0
        elif train_step_index == self.pruning_start:
            alpha = self.alpha_i_lambda_mix
        elif self.pruning_start < train_step_index < self.pruning_end:
            frac_of_total = 1 - (train_step_index - self.pruning_start) / (self.pruning_end - self.pruning_start)
            alpha = self.alpha_f_lambda_mix + (self.alpha_i_lambda_mix - self.alpha_f_lambda_mix) * frac_of_total
        elif train_step_index >= self.pruning_end:
            alpha = self.alpha_f_lambda_mix
        else:
            raise ValueError(f"Invalid train_step_index value: {train_step_index}")
        return alpha*self.lambda_mix
    
    def apply_prior_grad(self, model, train_step_index):
        sigma0, sigma1 = self.sigma0, self.sigma1
        lambda_mix = self.annealing_scheduler(train_step_index)
        c1 = math.log(lambda_mix) - math.log(1 - lambda_mix) + 0.5 * math.log(sigma0) - 0.5 * math.log(sigma1)
        c2 = 0.5 / sigma0 - 0.5 / sigma1
        prior_threshold = math.sqrt(math.log((1 - lambda_mix) / lambda_mix * math.sqrt(sigma1 / sigma0)) / (
                    0.5 / sigma0 - 0.5 / sigma1))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_prune_param(n):
                    p.grad.sub_(
                        p.mul(
                            p.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                            .mul((sigma0-sigma1)/(self.train_size*sigma0*sigma1))
                            .add((-1)/(self.train_size*sigma1))
                        )
                    )
        return prior_threshold, sigma0, sigma1, lambda_mix
    
    def calculate_mask_threshold(self, model, sparsity):
        is_dict = {}
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                is_dict[n] = p.detach().abs()
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * sparsity))[0].item()
        return mask_threshold, is_dict
    
    def create_pruning_mask(self, model, mask_threshold, is_dict):
        mask = {}
        for n, _ in model.named_parameters():
            if self.whether_prune_param(n):
                mask[n] = (is_dict[n] < mask_threshold)
        return mask
    
    def prune_with_threshold(self, model, sparsity):
        mask_threshold, is_dict = self.calculate_mask_threshold(model, sparsity)
        mask = self.create_pruning_mask(model, mask_threshold, is_dict)
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                p.detach().masked_fill_(mask[n], 0.0)
        return mask_threshold, mask
   
    def magnitude_pruning(self, model, train_step_index):
        sparsity, pruning_ind = self.sparsity_scheduler(train_step_index)
        if pruning_ind and sparsity > 0.0:
            if train_step_index == self.pruning_end:
                print ("Gradual pruning stage is over. Generate the final fixed mask...")
                mask_threshold, mask = self.prune_with_threshold(model, sparsity)
                self.final_fixed_mask = mask
            elif train_step_index > self.pruning_end:
                self.prune_with_mask(model, self.final_fixed_mask)
                mask = self.final_fixed_mask
                mask_threshold = 0.0
            else:
                mask_threshold, mask = self.prune_with_threshold(model, sparsity)
        else:
            mask_threshold = None
            mask = None
        return sparsity, mask_threshold, mask
    
    def sparsity_scheduler(self, train_step_index):
        pruning_ind = False
        if train_step_index < self.pruning_start:
            sparsity = 0.0
            pruning_ind = False
        elif train_step_index == self.pruning_start:
            sparsity = self.initial_sparsity
            pruning_ind = True
        elif self.pruning_start < train_step_index < self.pruning_end:
            frac_of_total = 1 - (train_step_index - self.pruning_start) / (self.pruning_end - self.pruning_start)
            sparsity = self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (frac_of_total ** 3)
            pruning_ind = True if train_step_index % self.pruning_interval == 0 else False
        elif train_step_index >= self.pruning_end:
            sparsity = self.final_sparsity
            pruning_ind = True
        else:
            raise ValueError(f"Invalid train_step_index value: {train_step_index}")
        return sparsity, pruning_ind
            
    def prune_with_mask(self, model, mask):
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                p.detach().masked_fill_(mask[n], 0.0)
    
    def count_param_below_prior_threshold(self, model, prior_threshold):
        n_param_below_prior_threshold = 0
        in_spike_mask = {}
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                in_spike_mask[n] = (p.detach().abs() < prior_threshold)
                n_param_below_prior_threshold += in_spike_mask[n].sum().item()
        return n_param_below_prior_threshold, in_spike_mask
    
    def get_magnitude_stat(self, model, which, mask=None):
        magnitude_vector = []
        for n, p in model.named_parameters():
            if which == "remaining_candidate":
                if self.whether_prune_param(n):
                    if mask is not None:
                        magnitude_vector.append(p.detach().abs()[~mask[n]].view(-1))
                    else:
                        magnitude_vector.append(p.detach().abs().view(-1))
            elif which == "remaining_non_candidate":
                if not self.whether_prune_param(n):
                    magnitude_vector.append(p.detach().abs().view(-1))
            elif which == "pruned_candidate":
                if self.whether_prune_param(n):
                    if mask is not None:
                        magnitude_vector.append(p.detach().abs()[mask[n]].view(-1))
                    else:
                        raise ValueError("mask must be provided for the pruned_candidate magnitude vector")
            else:
                raise ValueError("which must be one of the following: remaining_candidate, remaining_non_candidate, pruned_candidate")
        magnitude_vector = torch.cat(magnitude_vector)
        magnitude_stat = {}
        magnitude_stat["model/magnitude_avg"] = magnitude_vector.mean().item()
        magnitude_stat["model/magnitude_std"] = magnitude_vector.std().item()
        return magnitude_stat
    
    def print_pruning_modules(self, model):
        print ("List of model modules to be pruned:")
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                print (n)
                self.n_total_param_for_pruning += p.numel()
        print ("Total number of candidate parameters for pruning:", self.n_total_param_for_pruning)

    def match(self, event, state):
        return event in [Event.FIT_START, Event.AFTER_TRAIN_BATCH, Event.BATCH_END]

    def apply(self, event, state, logger):
        if event == Event.FIT_START:
            train_step_index = state.timestamp.batch.value
            # print the list of model modules to be pruned
            self.print_pruning_modules(state.model)
            # log the parameter's magnitude statistics of the pre-trained model
            if (self.log_interval is not None and
                logger is not None and
                train_step_index == 0):
                remaining_candidate_magnitude_stat = self.get_magnitude_stat(state.model, which="remaining_candidate")
                remaining_non_candidate_magnitude_stat = self.get_magnitude_stat(state.model, which="remaining_non_candidate")
                logger.log_metrics({"model/remaining_candidate_magnitude_avg": float(remaining_candidate_magnitude_stat["model/magnitude_avg"]),
                                    "model/remaining_candidate_magnitude_std": float(remaining_candidate_magnitude_stat["model/magnitude_std"]),
                                    "model/remaining_non_candidate_magnitude_avg": float(remaining_non_candidate_magnitude_stat["model/magnitude_avg"]),
                                    "model/remaining_non_candidate_magnitude_std": float(remaining_non_candidate_magnitude_stat["model/magnitude_std"])})
            # in case we resume training from a checkpoint after the gradual pruning stage, we need to generate the final fixed mask first
            if (train_step_index > self.pruning_end and 
                self.final_fixed_mask is None):
                print ("Generate the final fixed mask first...")
                _, mask = self.prune_with_threshold(state.model, self.final_sparsity)
                self.final_fixed_mask = mask
        elif event == Event.AFTER_TRAIN_BATCH:
            train_step_index = state.timestamp.batch.value
            # add prior gradients to the model during the gradual pruning stage
            if train_step_index <= self.pruning_end:
                prior_threshold, sigma0, sigma1, lambda_mix = self.apply_prior_grad(state.model, train_step_index)
                if logger is not None:
                    logger.log_metrics({"prior/sigma0": float(sigma0)})
                    logger.log_metrics({"prior/sigma1": float(sigma1)})
                    logger.log_metrics({"prior/lambda_mix": float(lambda_mix)})
                    logger.log_metrics({"prior/prior_threshold": float(prior_threshold)})
                self.current_prior_threshold = prior_threshold
        elif event == Event.BATCH_END:
            train_step_index = state.timestamp.batch.value - 1
            # log the count of parameters remaining in the high-penalty region (spike) from the last pruning step right before the next pruning step
            if (self.log_interval is not None and
                logger is not None and
                train_step_index > self.pruning_start and
                train_step_index <= self.pruning_end and
                train_step_index % self.log_interval == 0):
                n_param_below_prior_threshold, in_spike_mask = self.count_param_below_prior_threshold(state.model, self.current_prior_threshold)
                logger.log_metrics({"pruning/percent_remained_in_spike": float(n_param_below_prior_threshold/self.n_total_param_for_pruning)})
            # perform magnitude pruning
            sparsity, mask_threshold, mask = self.magnitude_pruning(state.model, train_step_index)
            if mask is not None:
                self.current_sparsity_mask = mask
            # log the current sparsity
            if logger is not None:
                logger.log_metrics({"pruning/sparsity": float(sparsity)})
                # if the current mask threshold is not None, log the current mask threshold
                if mask_threshold is not None:
                    logger.log_metrics({"pruning/mask_threshold": float(mask_threshold)})
            # log how mask corresponds to the final sparsity changes during the gradual pruning stage
            if (self.log_interval is not None and
                logger is not None and
                train_step_index >= self.pruning_start and
                train_step_index <= self.pruning_end):
                if train_step_index == self.pruning_start:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_sparsity)
                    self.after_initial_warmup_mask = self.create_pruning_mask(state.model, mask_threshold, is_dict)
                if train_step_index > self.pruning_start and train_step_index % self.log_interval == 0:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_sparsity)
                    updated_mask = self.create_pruning_mask(state.model, mask_threshold, is_dict)
                    if self.current_mask is not None:
                        n_diff = _count_mask_differences(self.current_mask, updated_mask)
                        logger.log_metrics({"pruning/n_mask_diff_wrt_current_mask": int(n_diff)})
                    if self.after_initial_warmup_mask is not None:
                        n_diff = _count_mask_differences(self.after_initial_warmup_mask, updated_mask)
                        logger.log_metrics({"pruning/n_mask_diff_wrt_after_initial_warmup_mask": int(n_diff)})
                    self.current_mask = updated_mask
            # log the parameter's magnitude statistics during training
            if (self.log_interval is not None and
                logger is not None):
                if train_step_index % self.log_interval == 0:
                    remaining_candidate_magnitude_stat = self.get_magnitude_stat(state.model, which="remaining_candidate", mask=mask)
                    remaining_non_candidate_magnitude_stat = self.get_magnitude_stat(state.model, which="remaining_non_candidate")
                    logger.log_metrics({"model/remaining_candidate_magnitude_avg": float(remaining_candidate_magnitude_stat["model/magnitude_avg"]),
                                        "model/remaining_candidate_magnitude_std": float(remaining_candidate_magnitude_stat["model/magnitude_std"]),
                                        "model/remaining_non_candidate_magnitude_avg": float(remaining_non_candidate_magnitude_stat["model/magnitude_avg"]),
                                        "model/remaining_non_candidate_magnitude_std": float(remaining_non_candidate_magnitude_stat["model/magnitude_std"])})
                elif (train_step_index + 1) % self.log_interval == 0 and self.current_sparsity_mask is not None:
                    pruned_candidate_magnitude_stat = self.get_magnitude_stat(state.model, which="pruned_candidate", mask=self.current_sparsity_mask)
                    logger.log_metrics({"model/pruned_candidate_magnitude_avg": float(pruned_candidate_magnitude_stat["model/magnitude_avg"]),
                                        "model/pruned_candidate_magnitude_std": float(pruned_candidate_magnitude_stat["model/magnitude_std"])})