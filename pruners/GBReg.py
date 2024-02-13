import torch
import re
import numpy as np
from composer.core import Algorithm, Event
from pruners.utils_composer import _convert_timestr_to_int

def _linear_scheduler(step, start, end, start_value, end_value):
    if step <= start:
        return start_value
    elif step >= end:
        return end_value
    else:
        if start_value == end_value:
            return start_value
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
            sigma0=1e-15,
            # initial factor value for sigma0
            alpha_i_sigma0=1.0,
            # final factor value for sigma0
            alpha_f_sigma0=1.0,
            # start step of annealing for sigma0
            anneal_start_sigma0=None,
             # end step of annealing for sigma0
            anneal_end_sigma0=None,
            # initial value of sigma1
            sigma1=0.05,
            # initial value of lambda_mix
            lambda_mix=1e-4,
            # initial factor value for lambda_mix
            alpha_i_lambda_mix=1.0,
            # final factor value for lambda_mix
            alpha_f_lambda_mix=1.0,
            # start step of annealing for lambda_mix
            anneal_start_lambda_mix=None,
            # end step of annealing for lambda_mix
            anneal_end_lambda_mix=None,
            # initial sparsity ratio, if set to less than 1.0, will prune the model to this ratio at the beginning of the graudal pruning stage
            initial_ratio=0.7,
            # target remaining ratio, i.e., final_ratio = 1 - the target sparsity
            final_ratio=0.1,
            # number of steps for the initial warmup stage
            initial_warmup_steps=0,
            # number of steps for the final warmup stage
            final_warmup_steps=0,
            # number of training steps between two consecutive pruning steps
            pruning_interval=10,
            # parmeter names that should not be considered for both pruning and regularization, matched by regular expression, if None, all parameters are considered
            non_mask_name=None,
            # gradient norm clipping threshold, used during the final warmup stage if not None
            clipping_threshold=1.0,
            # interval for logging the parameter's magnitude statistics during the initial warmup stage
            magnitude_stat_log_interval=None,
            # interval for logging the mask change during the gradual pruning stage
            mask_change_log_interval=None,
            # flag to log the count and percentage of parameters remaining in the high-penalty region (spike) after one optimization step post-pruning.
            log_spike_remainings=False
        ):

        self.after_initial_warmup_mask = None
        self.current_mask = None
        self.final_fixed_mask = None
        self.current_prior_threshold = None

        self.magnitude_stat_log_interval = magnitude_stat_log_interval
        self.mask_change_log_interval = mask_change_log_interval
        self.log_spike_remainings = log_spike_remainings

        self.train_size = train_size
        self.max_train_steps = max_train_steps

        self.clipping_threshold = clipping_threshold

        self.non_mask_name_pattern = re.compile("|".join(non_mask_name), re.IGNORECASE) if non_mask_name is not None else None

        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio

        pruning_start = initial_warmup_steps
        pruning_end = max_train_steps - final_warmup_steps

        assert pruning_start < pruning_end <= max_train_steps, (
            f"pruning_start: {pruning_start}, "
            f"pruning_end: {pruning_end}, "
            f"max_train_steps: {max_train_steps}. "
            "Condition pruning_start < pruning_end <= max_train_steps must be satisfied, but got False"
        )

        self.sigma0 = sigma0
        self.alpha_i_sigma0 = alpha_i_sigma0
        self.alpha_f_sigma0 = alpha_f_sigma0
        self.anneal_start_sigma0 = anneal_start_sigma0 if anneal_start_sigma0 is not None else pruning_start
        self.anneal_end_sigma0 = anneal_end_sigma0 if anneal_end_sigma0 is not None else pruning_end

        # we do not anneal sigma1
        self.sigma1 = sigma1

        self.lambda_mix = lambda_mix
        self.alpha_i_lambda_mix = alpha_i_lambda_mix
        self.alpha_f_lambda_mix = alpha_f_lambda_mix
        self.anneal_start_lambda_mix = anneal_start_lambda_mix if anneal_start_lambda_mix is not None else 0
        self.anneal_end_lambda_mix = anneal_end_lambda_mix if anneal_end_lambda_mix is not None else pruning_end
    
        self.pruning_start = pruning_start
        self.pruning_end = pruning_end
        self.pruning_steps = pruning_end - pruning_start
        self.pruning_interval = pruning_interval


    @classmethod
    def from_args(self, train_size, max_train_steps, train_dataloader_len, args):
        initial_warmup_steps = _convert_timestr_to_int(args.initial_warmup_steps, max_train_steps, train_dataloader_len)
        final_warmup_steps = _convert_timestr_to_int(args.final_warmup_steps, max_train_steps, train_dataloader_len)
        pruning_interval = _convert_timestr_to_int(args.pruning_interval, max_train_steps, train_dataloader_len)
        anneal_start_sigma0 = _convert_timestr_to_int(args.anneal_start_sigma0, max_train_steps, train_dataloader_len) if args.anneal_start_sigma0 is not None else None
        anneal_end_sigma0 = _convert_timestr_to_int(args.anneal_end_sigma0, max_train_steps, train_dataloader_len) if args.anneal_end_sigma0 is not None else None
        anneal_start_lambda_mix = _convert_timestr_to_int(args.anneal_start_lambda_mix, max_train_steps, train_dataloader_len) if args.anneal_start_lambda_mix is not None else None
        anneal_end_lambda_mix = _convert_timestr_to_int(args.anneal_end_lambda_mix, max_train_steps, train_dataloader_len) if args.anneal_end_lambda_mix is not None else None
        magnitude_stat_log_interval = _convert_timestr_to_int(args.magnitude_stat_log_interval, max_train_steps, train_dataloader_len) if args.magnitude_stat_log_interval is not None else None
        mask_change_log_interval = _convert_timestr_to_int(args.mask_change_log_interval, max_train_steps, train_dataloader_len) if args.mask_change_log_interval is not None else None
        return self(
            train_size=train_size,
            max_train_steps=max_train_steps,
            sigma0=args.sigma0,
            alpha_i_sigma0=args.alpha_i_sigma0,
            alpha_f_sigma0=args.alpha_f_sigma0,
            anneal_start_sigma0=anneal_start_sigma0,
            anneal_end_sigma0=anneal_end_sigma0,
            sigma1=args.sigma1,
            lambda_mix=args.lambda_mix,
            alpha_i_lambda_mix=args.alpha_i_lambda_mix,
            alpha_f_lambda_mix=args.alpha_f_lambda_mix,
            anneal_start_lambda_mix=anneal_start_lambda_mix,
            anneal_end_lambda_mix=anneal_end_lambda_mix,
            initial_ratio=args.initial_ratio,
            final_ratio=args.final_ratio,
            initial_warmup_steps=initial_warmup_steps,
            final_warmup_steps=final_warmup_steps,
            pruning_interval=pruning_interval,
            non_mask_name=args.non_mask_name,
            clipping_threshold=args.clipping_threshold,
            magnitude_stat_log_interval=magnitude_stat_log_interval,
            mask_change_log_interval=mask_change_log_interval,
            log_spike_remainings=args.log_spike_remainings
        )

    def whether_mask_param(self, n):
        if self.non_mask_name_pattern == None:
            return True
        else:
            return not bool(re.search(self.non_mask_name_pattern, n))
        
    def linear_prior_scheduler(self, train_step_index):
        lambda_mix_factor = _linear_scheduler(
            train_step_index, 
            self.anneal_start_lambda_mix, 
            self.anneal_end_lambda_mix, 
            self.alpha_i_lambda_mix, 
            self.alpha_f_lambda_mix
        )
        sigma0_factor = _linear_scheduler(
            train_step_index, 
            self.anneal_start_sigma0, 
            self.anneal_end_sigma0,
            self.alpha_i_sigma0, 
            self.alpha_f_sigma0
        )
        return sigma0_factor*self.sigma0, self.sigma1, lambda_mix_factor*self.lambda_mix

    def calculate_prior_grad_components(self, train_step_index):
        sigma0, sigma1, lambda_mix = self.linear_prior_scheduler(train_step_index)
        c1 = np.log(lambda_mix) - np.log(1 - lambda_mix) + 0.5 * np.log(sigma0) - 0.5 * np.log(sigma1)
        c2 = 0.5 / sigma0 - 0.5 / sigma1
        prior_threshold = np.sqrt(np.log((1 - lambda_mix) / lambda_mix * np.sqrt(sigma1 / sigma0)) / (
                    0.5 / sigma0 - 0.5 / sigma1))
        return c1, c2, prior_threshold, sigma0, sigma1, lambda_mix
    
    def add_prior_grad(self, model, train_step_index):
        c1, c2, prior_threshold, sigma0, sigma1, lambda_mix = self.calculate_prior_grad_components(train_step_index)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_param(n):
                    temp = p.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = temp.mul((sigma0-sigma1)/(self.train_size*sigma0*sigma1)).add((-1)/(self.train_size*sigma1))
                    p.grad -= p.mul(temp)
        return prior_threshold, sigma0, sigma1, lambda_mix
    
    def calculate_mask_threshold(self, model, ratio):
        is_dict = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_param(n):
                    is_dict[n] = p.abs().detach()
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - ratio)))[0].item()
        return mask_threshold, is_dict
    
    def create_mask(self, model, mask_threshold, is_dict):
        mask = {}
        for n, _ in model.named_parameters():
            if self.whether_mask_param(n):
                mask[n] = (is_dict[n] < mask_threshold)
        return mask
    
    def mask_with_threshold(self, model, ratio):
        mask_threshold, is_dict = self.calculate_mask_threshold(model, ratio)
        mask = self.create_mask(model, mask_threshold, is_dict)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_param(n):
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
        elif train_step_index >= self.pruning_end:
            ratio = self.final_ratio
            mask_ind = True
        else:
            mul_coeff = 1 - (train_step_index - self.pruning_start) / (self.pruning_steps)
            ratio = self.final_ratio + (self.initial_ratio - self.final_ratio) * (mul_coeff ** 3)
            mask_ind = True if train_step_index % self.pruning_interval == 0 else False
        return ratio, mask_ind
    
    def zero_masked_param_grad(self, model, mask):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_param(n):
                    p.grad.masked_fill_(mask[n], 0.0)
    
    def gradient_clipping(self, model, mask):
        self.zero_masked_param_grad(model, mask)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clipping_threshold).item()
        return grad_norm

    def calculate_relative_sparsity(self, model):
        n_param = 0
        n_masked_param = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_param(n):
                    n_param += p.numel()
                    n_masked_param += p.eq(0.0).sum().item()
        return n_masked_param/n_param
    
    def prune_with_fixed_mask(self, model, mask):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_param(n):
                    p.data.masked_fill_(mask[n], 0.0)
    
    def magnitude_stat(self, model):
        magnitude_vector = []
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_param(n):
                    magnitude_vector.append(p.abs().detach().view(-1))
        magnitude_vector = torch.cat(magnitude_vector)
        magnitude_stat = {}
        magnitude_stat["avg"] = float(magnitude_vector.mean())
        magnitude_stat["std"] = float(magnitude_vector.std())
        return magnitude_stat
    
    def count_params_below_prior_threshold(self, model, prior_threshold):
        n_param_below_prior_threshold = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_param(n):
                    n_param_below_prior_threshold += (p.abs() < prior_threshold).sum().item()
        return n_param_below_prior_threshold
    
    def print_pruning_modules(self, model):
        print ("list of model modules to be pruned:")
        for n, _ in model.named_parameters():
            if self.whether_mask_param(n):
                print (n)

    def match(self, event, state):
        return event in [Event.FIT_START, Event.AFTER_TRAIN_BATCH, Event.BATCH_END]

    def apply(self, event, state, logger):
        if event == Event.FIT_START:
            # print the list of model modules to be pruned
            self.print_pruning_modules(state.model)
            # log the parameter's magnitude statistics of the pre-trained model
            if self.magnitude_stat_log_interval is not None:
                magnitude_stat = self.magnitude_stat(state.model)
                logger.log_metrics(magnitude_stat)
            # in case we resume training from a checkpoint after the gradual pruning stage, we need to generate the final fixed mask first
            if state.timestamp.batch.value > self.pruning_end and self.final_fixed_mask is None:
                print ("generate the final fixed mask first...")
                mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                self.final_fixed_mask = self.create_mask(state.model, mask_threshold, is_dict)
        elif event == Event.AFTER_TRAIN_BATCH:
            # add prior gradients to the model during the gradual pruning stage
            if state.timestamp.batch.value <= self.pruning_end:
                prior_threshold, sigma0, sigma1, lambda_mix = self.add_prior_grad(state.model, state.timestamp.batch.value)
                logger.log_metrics({"sigma0": float(sigma0)})
                logger.log_metrics({"sigma1": float(sigma1)})
                logger.log_metrics({"lambda_mix": float(lambda_mix)})
                logger.log_metrics({"prior_threshold": float(prior_threshold)})
                self.current_prior_threshold = prior_threshold
            # perform gradient clipping during the final warmup stage
            if state.timestamp.batch.value > self.pruning_end and self.clipping_threshold is not None:
                grad_norm = self.gradient_clipping(state.model, self.final_fixed_mask)
                logger.log_metrics({"grad_norm": float(grad_norm)})
        elif event == Event.BATCH_END:
            # perform magnitude pruning
            ratio, mask_threshold, mask = self.magnitude_pruning(state.model, state.timestamp.batch.value)
            # log the current remaining ratio
            logger.log_metrics({"remaining_ratio": float(ratio)})
            # if the current mask threshold is not None, log the current mask threshold
            if mask_threshold is not None:
                logger.log_metrics({"mask_threshold": float(mask_threshold)})
            # log the parameter's magnitude statistics
            if (self.magnitude_stat_log_interval is not None and
                state.timestamp.batch.value < self.pruning_start and
                state.timestamp.batch.value % self.magnitude_stat_log_interval == 0):
                magnitude_stat = self.magnitude_stat(state.model)
                logger.log_metrics(magnitude_stat)
            # log how mask corresponds to the final ratio changes during the gradual pruning stage
            if (self.mask_change_log_interval is not None and 
                state.timestamp.batch.value >= self.pruning_start and
                state.timestamp.batch.value <= self.pruning_end):
                if state.timestamp.batch.value == self.pruning_start:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                    self.after_initial_warmup_mask = self.create_mask(state.model, mask_threshold, is_dict)
                if state.timestamp.batch.value > self.pruning_start and state.timestamp.batch.value % self.mask_change_log_interval == 0:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                    updated_mask = self.create_mask(state.model, mask_threshold, is_dict)
                    if self.current_mask is not None:
                        n_diff = _count_mask_differences(self.current_mask, updated_mask)
                        logger.log_metrics({"n_mask_diff_wrt_current_mask": int(n_diff)})
                    if self.after_initial_warmup_mask is not None:
                        n_diff = _count_mask_differences(self.after_initial_warmup_mask, updated_mask)
                        logger.log_metrics({"n_mask_diff_wrt_initial_warmup": int(n_diff)})
                    self.current_mask = updated_mask
            # log the count and percentage of parameters remaining in the high-penalty region (spike) after one optimization step post-pruning
            if (self.log_spike_remainings and 
                state.timestamp.batch.value > self.pruning_start and
                state.timestamp.batch.value < self.pruning_end and
                state.timestamp.batch.value-1 % self.pruning_interval == 0):
                n_param_below_prior_threshold = self.count_params_below_prior_threshold(state.model, self.current_prior_threshold)
                logger.log_metrics({"n_param_below_prior_threshold": int(n_param_below_prior_threshold)})
