import torch
import math
import re
from composer.core import Algorithm, Event
from pruners.utils_composer import _convert_timestr_to_int

def _linear_scheduler(step, start, end, start_value, end_value):
    if start_value == end_value:
        return start_value
    elif step <= start:
        return start_value
    elif step >= end:
        return end_value
    else:
        frac_of_total = min(1.0, (step - start) / (end - start))
        current_factor = start_value + frac_of_total * (end_value - start_value)
        return current_factor
    
def _count_mask_differences(mask1, mask2):
    n_same = 0
    n_total = 0
    for key in mask1.keys():
        n_total += mask1[key].numel()
        n_same += torch.eq(mask1[key], mask2[key]).sum().item()
    n_diff = n_total - n_same
    return n_diff

class GBReg(Algorithm):
    def __init__(
            self,
            # total number of training samples
            train_size,
            # total number of training steps
            max_train_steps,
            # initial value of sigma0
            sigma0=1e-10,
            # final factor value of sigma0
            alpha_f_sigma0=1.0,
            # the training step to start annealing sigma0, if None, will start from the beginning of the gradual pruning stage
            anneal_start_sigma0=None,
            # the training step to end annealing sigma0, if None, will end at the end of the gradual pruning stage
            anneal_end_sigma0=None,
            # initial value of sigma1
            sigma1=0.05,
            # final factor value of sigma1
            alpha_f_sigma1=1.0,
            # the training step to start annealing sigma1, if None, will start from the beginning of the gradual pruning stage
            anneal_start_sigma1=None,
            # the training step to end annealing sigma1, if None, will end at the end of the gradual pruning stage
            anneal_end_sigma1=None,
            # initial value of lambda_mix
            lambda_mix=1e-3,
            # final factor value of lambda_mix
            alpha_f_lambda_mix=1.0,
            # the training step to start annealing lambda_mix, if None, will start from the beginning of the gradual pruning stage
            anneal_start_lambda_mix=None,
            # the training step to end annealing lambda_mix, if None, will end at the end of the gradual pruning stage
            anneal_end_lambda_mix=None,
            # initial remaining ratio, if set to less than 1.0, will prune the model to this ratio at the beginning of the graudal pruning stage
            initial_ratio=1.0,
            # target remaining ratio, i.e., final_ratio = 1 - the target sparsity
            final_ratio=0.1,
            # number of steps for the initial warmup stage
            initial_warmup_steps=0,
            # number of steps for the final warmup stage
            final_warmup_steps=0,
            # number of training steps between two consecutive pruning steps
            pruning_interval=10,
            # parmeter names that should not be considered for both pruning and regularization, matched by regular expression, if None, all parameters are considered
            pruning_params=None,
            # gradient norm clipping threshold, for more details, please refer to both the apply and the gradient_clipping functions
            clipping_threshold=None,
            # number of steps for the sparse fine-tuning stage, i.e., fine-tuning the pruned model with fixed mask
            sparse_finetune_steps=0,
            # interval for logging all research-related metrics, if None, no logging will be performed
            log_interval=None,
        ):

        self.num_total_params_for_pruning = None
        self.after_initial_warmup_mask = None
        self.current_mask = None
        self.final_fixed_mask = None
        self.current_prior_threshold = None
        self.current_ratio_mask = None

        self.train_size = train_size
        self.max_train_steps = max_train_steps

        self.clipping_threshold = clipping_threshold

        self.pruning_params = re.compile("|".join(pruning_params), re.IGNORECASE) if pruning_params is not None else None

        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio

        pruning_start = initial_warmup_steps
        final_warmup_start = max_train_steps - sparse_finetune_steps - final_warmup_steps
        pruning_end = max_train_steps - sparse_finetune_steps

        assert pruning_start < final_warmup_start <= pruning_end <= max_train_steps, (
            f"pruning_start: {pruning_start}, "
            f"final_warmup_start: {final_warmup_start}, "
            f"pruning_end: {pruning_end}, "
            f"max_train_steps: {max_train_steps}. "
            "Condition pruning_start < final_warmup_start <= pruning_end <= max_train_steps must be satisfied, but got False"
        )

        if log_interval is not None and log_interval % pruning_interval != 0:
            print (
                f"log_interval: {log_interval}, "
                f"pruning_interval: {pruning_interval}. "
                "When log_interval is not None, log_interval must be divisible by pruning_interval, "
                "otherwise, only some of the metrics will be logged. "
                "For more details, please refer to the apply function."
            )
        
        self.log_interval = log_interval

        self.sigma0 = sigma0
        self.alpha_f_sigma0 = alpha_f_sigma0
        self.anneal_start_sigma0 = anneal_start_sigma0 if anneal_start_sigma0 is not None else pruning_start
        self.anneal_end_sigma0 = anneal_end_sigma0 if anneal_end_sigma0 is not None else final_warmup_start

        self.sigma1 = sigma1
        self.alpha_f_sigma1 = alpha_f_sigma1
        self.anneal_start_sigma1 = anneal_start_sigma1 if anneal_start_sigma1 is not None else pruning_start
        self.anneal_end_sigma1 = anneal_end_sigma1 if anneal_end_sigma1 is not None else final_warmup_start

        self.lambda_mix = lambda_mix
        self.alpha_f_lambda_mix = alpha_f_lambda_mix
        self.anneal_start_lambda_mix = anneal_start_lambda_mix if anneal_start_lambda_mix is not None else pruning_start
        self.anneal_end_lambda_mix = anneal_end_lambda_mix if anneal_end_lambda_mix is not None else final_warmup_start
    
        self.pruning_start = pruning_start
        self.final_warmup_start = final_warmup_start
        self.pruning_end = pruning_end
        self.remaining_ratio_scheduler_steps = final_warmup_start - pruning_start
        self.pruning_interval = pruning_interval

    # initialize the algorithm from the command line arguments
    @classmethod
    def from_args(self, train_size, max_train_steps, train_dataloader_len, args):
        initial_warmup_steps = _convert_timestr_to_int(args.initial_warmup_steps, max_train_steps, train_dataloader_len)
        final_warmup_steps = _convert_timestr_to_int(args.final_warmup_steps, max_train_steps, train_dataloader_len)
        sparse_finetune_steps = _convert_timestr_to_int(args.sparse_finetune_steps, max_train_steps, train_dataloader_len)
        pruning_interval = _convert_timestr_to_int(args.pruning_interval, max_train_steps, train_dataloader_len)
        log_interval = _convert_timestr_to_int(args.log_interval, max_train_steps, train_dataloader_len) if args.log_interval is not None else None
        anneal_start_sigma0 = _convert_timestr_to_int(args.anneal_start_sigma0, max_train_steps, train_dataloader_len) if args.anneal_start_sigma0 is not None else None
        anneal_end_sigma0 = _convert_timestr_to_int(args.anneal_end_sigma0, max_train_steps, train_dataloader_len) if args.anneal_end_sigma0 is not None else None
        anneal_start_sigma1 = _convert_timestr_to_int(args.anneal_start_sigma1, max_train_steps, train_dataloader_len) if args.anneal_start_sigma1 is not None else None
        anneal_end_sigma1 = _convert_timestr_to_int(args.anneal_end_sigma1, max_train_steps, train_dataloader_len) if args.anneal_end_sigma1 is not None else None
        anneal_start_lambda_mix = _convert_timestr_to_int(args.anneal_start_lambda_mix, max_train_steps, train_dataloader_len) if args.anneal_start_lambda_mix is not None else None
        anneal_end_lambda_mix = _convert_timestr_to_int(args.anneal_end_lambda_mix, max_train_steps, train_dataloader_len) if args.anneal_end_lambda_mix is not None else None
        return self(
            train_size=train_size,
            max_train_steps=max_train_steps,
            sigma0=args.sigma0,
            alpha_f_sigma0=args.alpha_f_sigma0,
            anneal_start_sigma0=anneal_start_sigma0,
            anneal_end_sigma0=anneal_end_sigma0,
            sigma1=args.sigma1,
            alpha_f_sigma1=args.alpha_f_sigma1,
            anneal_start_sigma1=anneal_start_sigma1,
            anneal_end_sigma1=anneal_end_sigma1,
            lambda_mix=args.lambda_mix,
            alpha_f_lambda_mix=args.alpha_f_lambda_mix,
            anneal_start_lambda_mix=anneal_start_lambda_mix,
            anneal_end_lambda_mix=anneal_end_lambda_mix,
            initial_ratio=args.initial_ratio,
            final_ratio=args.final_ratio,
            initial_warmup_steps=initial_warmup_steps,
            final_warmup_steps=final_warmup_steps,
            sparse_finetune_steps=sparse_finetune_steps,
            pruning_interval=pruning_interval,
            pruning_params=args.pruning_params,
            clipping_threshold=args.clipping_threshold,
            log_interval=log_interval,
        )

    def whether_prune_param(self, n):
        if self.pruning_params == None:
            return True
        else:
            return bool(re.search(self.pruning_params, n))
        
    def prior_annealing_scheduler(self, train_step_index):
        sigma0_factor = _linear_scheduler(
            train_step_index, 
            self.anneal_start_sigma0,
            self.anneal_end_sigma0,
            1.0,
            self.alpha_f_sigma0
        )
        sigma1_factor = _linear_scheduler(
            train_step_index, 
            self.anneal_start_sigma1, 
            self.anneal_end_sigma1, 
            1.0,
            self.alpha_f_sigma1
        )
        lambda_mix_factor = _linear_scheduler(
            train_step_index, 
            self.anneal_start_lambda_mix, 
            self.anneal_end_lambda_mix, 
            1.0,
            self.alpha_f_lambda_mix
        )
        return sigma0_factor*self.sigma0, sigma1_factor*self.sigma1, lambda_mix_factor*self.lambda_mix

    def compute_prior_grad_components(self, train_step_index):
        sigma0, sigma1, lambda_mix = self.prior_annealing_scheduler(train_step_index)
        c1 = math.log(lambda_mix) - math.log(1 - lambda_mix) + 0.5 * math.log(sigma0) - 0.5 * math.log(sigma1)
        c2 = 0.5 / sigma0 - 0.5 / sigma1
        prior_threshold = math.sqrt(math.log((1 - lambda_mix) / lambda_mix * math.sqrt(sigma1 / sigma0)) / (
                    0.5 / sigma0 - 0.5 / sigma1))
        return c1, c2, prior_threshold, sigma0, sigma1, lambda_mix
    
    def apply_prior_grad(self, model, train_step_index):
        c1, c2, prior_threshold, sigma0, sigma1, lambda_mix = self.compute_prior_grad_components(train_step_index)
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
    
    def calculate_mask_threshold(self, model, ratio):
        # is stands for importance score, in this case, the absolute value of the parameter
        is_dict = {}
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                is_dict[n] = p.detach().abs()
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - ratio)))[0].item()
        return mask_threshold, is_dict
    
    def create_mask(self, model, mask_threshold, is_dict):
        mask = {}
        for n, _ in model.named_parameters():
            if self.whether_prune_param(n):
                mask[n] = (is_dict[n] < mask_threshold)
        return mask
    
    def mask_with_threshold(self, model, ratio):
        mask_threshold, is_dict = self.calculate_mask_threshold(model, ratio)
        mask = self.create_mask(model, mask_threshold, is_dict)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_prune_param(n):
                    p.masked_fill_(mask[n], 0.0)
        return mask_threshold, mask
   
    def magnitude_pruning(self, model, train_step_index):
        ratio, mask_ind = self.remaining_ratio_scheduler(train_step_index)
        if mask_ind and ratio < 1.0:
            if train_step_index == self.pruning_end:
                mask_threshold, mask = self.mask_with_threshold(model, ratio)
                self.final_fixed_mask = mask
            elif train_step_index > self.pruning_end:
                self.prune_with_fixed_mask(model, self.final_fixed_mask)
                mask = self.final_fixed_mask
                mask_threshold = 0.0
            else:
                mask_threshold, mask = self.mask_with_threshold(model, ratio)
        else:
            mask_threshold = None
            mask = None
        return ratio, mask_threshold, mask
    
    def remaining_ratio_scheduler(self, train_step_index):
        mask_ind = False
        if train_step_index < self.pruning_start:
            ratio = 1.0
            mask_ind = False
        elif train_step_index == self.pruning_start:
            ratio = self.initial_ratio
            mask_ind = True
        elif self.pruning_start < train_step_index < self.final_warmup_start:
            mul_coeff = 1 - (train_step_index - self.pruning_start) / (self.remaining_ratio_scheduler_steps)
            ratio = self.final_ratio + (self.initial_ratio - self.final_ratio) * (mul_coeff ** 3)
            mask_ind = True if train_step_index % self.pruning_interval == 0 else False
        elif self.final_warmup_start <= train_step_index < self.pruning_end:
            ratio = self.final_ratio
            mask_ind = True #if train_step_index % self.pruning_interval == 0 else False
        elif train_step_index >= self.pruning_end:
            ratio = self.final_ratio
            mask_ind = True
        return ratio, mask_ind
    
    def gradient_clipping(self, model, mask):
        grads = []
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                grads.append(p.grad.detach()[~mask[n]].view(-1))
            else:
                grads.append(p.grad.detach().view(-1))
        total_norm = torch.cat(grads).norm(2).item()
        clip_coef = self.clipping_threshold / (total_norm + 1e-6)
        clip_coef = min(1, clip_coef)
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                p.grad.detach()[~mask[n]] = p.grad.detach()[~mask[n]] * clip_coef
            else:
                p.grad.detach().mul_(clip_coef)
        return total_norm
            
    def prune_with_fixed_mask(self, model, mask):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_prune_param(n):
                    p.masked_fill_(mask[n], 0.0)
    
    def magnitude_stat(self, model, mask=None):
        magnitude_vector = []
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                if mask is not None:
                    magnitude_vector.append(p.detach().abs()[~mask[n]].view(-1))
                else:
                    magnitude_vector.append(p.detach().abs().view(-1))
        magnitude_vector = torch.cat(magnitude_vector)
        magnitude_stat = {}
        magnitude_stat["avg"] = float(magnitude_vector.mean())
        magnitude_stat["std"] = float(magnitude_vector.std())
        return magnitude_stat
    
    def count_params_below_prior_threshold(self, model, prior_threshold):
        n_param_below_prior_threshold = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_prune_param(n):
                    n_param_below_prior_threshold += (p.abs() < prior_threshold).sum().item()
        return n_param_below_prior_threshold
    
    def get_grad_norms(self, model, mask=None):
        candidate_grads = []
        other_grads = []
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_prune_param(n):
                    if mask is not None:
                        candidate_grads.append(p.grad.detach()[~mask[n]].view(-1))
                    else:
                        candidate_grads.append(p.grad.detach().view(-1))
                else:
                    other_grads.append(p.grad.detach().view(-1))
            candidate_grad_norm = torch.cat(candidate_grads).norm(2).item()
            other_grad_norm = torch.cat(other_grads).norm(2).item()
            all_grad_norm = (candidate_grad_norm**2+other_grad_norm**2)**0.5
        return candidate_grad_norm, other_grad_norm, all_grad_norm
    
    def print_pruning_modules(self, model):
        print ("List of model modules to be pruned:")
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                print (n)
                self.num_total_params_for_pruning += p.numel()
        print ("Total number of candidate parameters for pruning:", self.num_total_params_for_pruning)

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
                magnitude_stat = self.magnitude_stat(state.model)
                logger.log_metrics({"model/magnitude_mean": magnitude_stat["avg"],
                                    "model/magnitude_std":  magnitude_stat["std"]})
            # in case we resume training from a checkpoint after the gradual pruning stage, we need to generate the final fixed mask first
            if (train_step_index > self.pruning_end and 
                self.final_fixed_mask is None):
                print ("Generate the final fixed mask first...")
                mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                self.final_fixed_mask = self.create_mask(state.model, mask_threshold, is_dict)
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
            # perform gradient clipping during the final warmup stage
            if (train_step_index > self.pruning_start and
                self.clipping_threshold is not None and
                self.current_ratio_mask is not None):
                self.gradient_clipping(state.model, self.current_ratio_mask)
        elif event == Event.BATCH_END:
            train_step_index = state.timestamp.batch.value - 1
            # log the count of parameters remaining in the high-penalty region (spike) from the last pruning step right before the next pruning step
            if (self.log_interval is not None and
                logger is not None and
                train_step_index> self.pruning_start and
                train_step_index <= self.pruning_end and
                train_step_index % self.pruning_interval == 0):
                n_param_below_prior_threshold = self.count_params_below_prior_threshold(state.model, self.current_prior_threshold)
                logger.log_metrics({"model/n_param_remained": int(n_param_below_prior_threshold)})
                logger.log_metrics({"model/percent_remained": float(n_param_below_prior_threshold/self.num_total_params_for_pruning)})
            # perform magnitude pruning
            ratio, mask_threshold, mask = self.magnitude_pruning(state.model, train_step_index)
            if mask is not None:
                self.current_ratio_mask = mask
            # log the current remaining ratio
            if logger is not None:
                logger.log_metrics({"model/remaining_ratio": float(ratio)})
                logger.log_metrics({"model/sparsity": float(1 - ratio)})
                # if the current mask threshold is not None, log the current mask threshold
                if mask_threshold is not None:
                    logger.log_metrics({"model/mask_threshold": float(mask_threshold)})
            # log how mask corresponds to the final ratio changes during the gradual pruning stage
            if (self.log_interval is not None and
                logger is not None and
                train_step_index >= self.pruning_start and
                train_step_index <= self.pruning_end):
                if train_step_index == self.pruning_start:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                    self.after_initial_warmup_mask = self.create_mask(state.model, mask_threshold, is_dict)
                if train_step_index> self.pruning_start and train_step_index % self.log_interval == 0:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                    updated_mask = self.create_mask(state.model, mask_threshold, is_dict)
                    if self.current_mask is not None:
                        n_diff = _count_mask_differences(self.current_mask, updated_mask)
                        logger.log_metrics({"model/n_mask_diff_wrt_current_mask": int(n_diff)})
                    if self.after_initial_warmup_mask is not None:
                        n_diff = _count_mask_differences(self.after_initial_warmup_mask, updated_mask)
                        logger.log_metrics({"model/n_mask_diff_wrt_initial_warmup_mask": int(n_diff)})
                    self.current_mask = updated_mask
            # log the parameter's magnitude statistics during the initial warmup stage
            if (self.log_interval is not None and
                logger is not None and
                train_step_index < self.pruning_start and
                train_step_index % self.log_interval == 0
                ):
                magnitude_stat = self.magnitude_stat(state.model)
                logger.log_metrics({"model/magnitude_mean": magnitude_stat["avg"],
                                    "model/magnitude_std":  magnitude_stat["std"]})
            # log the remaining parameter's magnitude statistics during the gradual pruning stage and the final warmup stage
            if (self.log_interval is not None and
                logger is not None and
                train_step_index >= self.pruning_start and
                train_step_index % self.log_interval == 0
                ):
                if train_step_index <= self.pruning_end and mask is not None:
                    magnitude_stat = self.magnitude_stat(state.model, mask)
                    logger.log_metrics({"model/remaining_magnitude_mean": magnitude_stat["avg"],
                                        "model/remaining_magnitude_std":  magnitude_stat["std"]})
                elif train_step_index > self.pruning_end and self.final_fixed_mask is not None:
                    magnitude_stat = self.magnitude_stat(state.model, self.final_fixed_mask)
                    logger.log_metrics({"model/remaining_magnitude_mean": magnitude_stat["avg"],
                                        "model/remaining_magnitude_std":  magnitude_stat["std"]})
            # log the parameter's gradient norm
            if (self.log_interval is not None and
                logger is not None and
                train_step_index % self.log_interval == 0
                ):
                if train_step_index <= self.pruning_start:
                    remaining_candidate_grad_norm, other_grad_norm, all_grad_norm = self.get_grad_norms(state.model)
                elif train_step_index > self.pruning_start and mask is not None:
                    remaining_candidate_grad_norm, other_grad_norm, all_grad_norm = self.get_grad_norms(state.model, mask)
                logger.log_metrics({"model/remaining_candidate_grad_norm": float(remaining_candidate_grad_norm)})
                logger.log_metrics({"model/other_grad_norm": float(other_grad_norm)})
                logger.log_metrics({"model/all_grad_norm": float(all_grad_norm)})
