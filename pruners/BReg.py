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

class BReg(Algorithm):
    def __init__(
            self,
            train_size,
            max_train_steps,
            sigma0=1e-13,
            alpha_i_sigma0=1.0, 
            alpha_f_sigma0=1.0,
            sigma1=0.05,
            alpha_i_sigma1=1.0, 
            alpha_f_sigma1=1.0,
            lambda_mix=1e-3,
            alpha_i_lambda=1.0, 
            alpha_f_lambda=0.01,
            anneal_start=None, 
            anneal_end=None,
            final_ratio=0.1, 
            initial_ratio=1.0,
            initial_warmup=0, 
            final_warmup=0,
            deltaT=10,
            deltaT_cooldown=1,
            sparse_fine_tune=0,
            masking_value=0.0, 
            non_mask_name=None, 
            non_prior_name=None,
            # Sparse fine-tuning parameters for optional sparse fine-tuning stage
            clipping_threshold=None,
            # whethter use the inital prior regularization (smaller) during the cool down stage
            init_prior_in_cooldown=False,
            # the interval to print the parameter statistics
            log_param_stat_interval=None,
        ):
        
        self.clipping_threshold = clipping_threshold
        self.init_prior_in_cooldown = init_prior_in_cooldown
        self.log_param_stat_interval = log_param_stat_interval

        self.final_ratio_mask = None
        self.train_size = train_size
        self.max_train_steps = max_train_steps
        self.masking_value = masking_value
        self.non_mask_name_pattern = re.compile("|".join(non_mask_name), re.IGNORECASE) if non_mask_name is not None else None
        self.non_prior_name_pattern = re.compile("|".join(non_prior_name), re.IGNORECASE) if non_prior_name is not None else None

        self.sigma0 = sigma0
        self.alpha_i_sigma0 = alpha_i_sigma0
        self.alpha_f_sigma0 = alpha_f_sigma0

        self.sigma1 = sigma1
        self.alpha_i_sigma1 = alpha_i_sigma1
        self.alpha_f_sigma1 = alpha_f_sigma1

        self.lambda_mix = lambda_mix
        self.alpha_i_lambda = alpha_i_lambda
        self.alpha_f_lambda = alpha_f_lambda

        self.final_ratio = final_ratio
        self.initial_ratio = initial_ratio
        self.initial_warmup = initial_warmup
        self.final_warmup = final_warmup
        self.deltaT = deltaT
        self.deltaT_cooldown = deltaT_cooldown

        self.sparse_fine_tune = sparse_fine_tune
        
        cubic_prune_start = initial_warmup
        cubic_prune_cool_down_start = max_train_steps - sparse_fine_tune - final_warmup
        cubic_prune_end = max_train_steps - sparse_fine_tune

        self.anneal_start = anneal_start if anneal_start is not None else cubic_prune_start
        self.anneal_end = anneal_end if anneal_end is not None else cubic_prune_cool_down_start

        if not (cubic_prune_start < cubic_prune_cool_down_start <= cubic_prune_end <= max_train_steps):
            print ("cubic_prune_start:", cubic_prune_start)
            print ("cubic_prune_cool_down:", cubic_prune_cool_down_start)
            print ("cubic_prune_end:", cubic_prune_end)
            print ("max_train_steps:", max_train_steps)
            raise ValueError("cubic_prune_start < cubic_prune_cool_down_start <= cubic_prune_end <= max_train_steps must be satisfied, but got False")
    
        self.ratio_scheduler_steps = cubic_prune_cool_down_start - cubic_prune_start
        self.cubic_prune_start = cubic_prune_start
        self.cubic_prune_end = cubic_prune_end
        self.cubic_prune_cool_down_start = cubic_prune_cool_down_start


    @classmethod
    def from_args(self, train_size, max_train_steps, train_dataloader_len, args):
        initial_warmup = _convert_timestr_to_int(args.initial_warmup, max_train_steps, train_dataloader_len)
        final_warmup = _convert_timestr_to_int(args.final_warmup, max_train_steps, train_dataloader_len)
        deltaT = _convert_timestr_to_int(args.deltaT, max_train_steps, train_dataloader_len)
        deltaT_cooldown = _convert_timestr_to_int(args.deltaT_cooldown, max_train_steps, train_dataloader_len)
        sparse_fine_tune = _convert_timestr_to_int(args.sparse_fine_tune, max_train_steps, train_dataloader_len)
        anneal_start = _convert_timestr_to_int(args.anneal_start, max_train_steps, train_dataloader_len) if args.anneal_start is not None else None
        anneal_end = _convert_timestr_to_int(args.anneal_end, max_train_steps, train_dataloader_len) if args.anneal_end is not None else None
        log_param_stat_interval = _convert_timestr_to_int(args.log_param_stat_interval, max_train_steps, train_dataloader_len) if args.log_param_stat_interval is not None else None
        return self(
            train_size, 
            max_train_steps,
            sigma0=args.sigma0, 
            alpha_i_sigma0=args.alpha_i_sigma0,
            alpha_f_sigma0=args.alpha_f_sigma0,
            sigma1=args.sigma1,
            alpha_i_sigma1=args.alpha_i_sigma1,
            alpha_f_sigma1=args.alpha_f_sigma1,
            lambda_mix=args.lambda_mix,
            alpha_i_lambda=args.alpha_i_lambda, 
            alpha_f_lambda=args.alpha_f_lambda,
            anneal_start=anneal_start, 
            anneal_end=anneal_end,
            final_ratio=args.final_ratio, 
            initial_ratio=args.initial_ratio,
            initial_warmup=initial_warmup, 
            final_warmup=final_warmup, 
            deltaT=deltaT,
            deltaT_cooldown=deltaT_cooldown,
            sparse_fine_tune=sparse_fine_tune,
            masking_value=args.masking_value, 
            non_mask_name=args.non_mask_name, 
            non_prior_name=args.non_prior_name,
            clipping_threshold=args.clipping_threshold,
            init_prior_in_cooldown=args.init_prior_in_cooldown,
            log_param_stat_interval=log_param_stat_interval,
        )
    
    def linear_prior_scheduler(self, train_step_index):
        if train_step_index >= self.cubic_prune_cool_down_start and self.init_prior_in_cooldown:
            lambda_mix_factor = self.alpha_i_lambda
            sigma0_factor = self.alpha_i_sigma0
            sigma1_factor = self.alpha_i_sigma1
        else:
            lambda_mix_factor = _linear_scheduler(train_step_index, self.anneal_start, self.anneal_end, self.alpha_i_lambda, self.alpha_f_lambda)
            sigma0_factor = _linear_scheduler(train_step_index, self.anneal_start, self.anneal_end, self.alpha_i_sigma0, self.alpha_f_sigma0)
            sigma1_factor = _linear_scheduler(train_step_index, self.anneal_start, self.anneal_end, self.alpha_i_sigma1, self.alpha_f_sigma1)
        return sigma0_factor*self.sigma0, sigma1_factor*self.sigma1, lambda_mix_factor*self.lambda_mix

    def whether_mask_para(self, n):
        if self.non_mask_name_pattern == None:
            return True
        else:
            return not bool(re.search(self.non_mask_name_pattern, n))
        
    def whether_penalize_para(self, n):
        if self.non_prior_name_pattern == None:
            return True
        else:
            return not bool(re.search(self.non_prior_name_pattern, n))

    def calculate_prior_grad_components(self, train_step_index):
        sigma0, sigma1, lambda_mix = self.linear_prior_scheduler(train_step_index)
        c1 = np.log(lambda_mix) - np.log(1 - lambda_mix) + 0.5 * np.log(sigma0) - 0.5 * np.log(sigma1)
        c2 = 0.5 / sigma0 - 0.5 / sigma1
        prior_threshold = np.sqrt(np.log((1 - lambda_mix) / lambda_mix * np.sqrt(sigma1 / sigma0)) / (
                    0.5 / sigma0 - 0.5 / sigma1))
        return c1, c2, prior_threshold, sigma0, sigma1, lambda_mix
    
    def add_prior_grad(self, model, train_step_index):
        # Add prior gradients to the model during the gradual cubic pruning stage
        c1, c2, prior_threshold, sigma0, sigma1, lambda_mix = self.calculate_prior_grad_components(train_step_index)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_penalize_para(n):
                    temp = p.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = temp.mul((sigma0-sigma1)/(self.train_size*sigma0*sigma1)).add((-1)/(self.train_size*sigma1))
                    p.grad -= p.mul(temp)
        return prior_threshold, sigma0, sigma1, lambda_mix
    
    def mask_with_threshold(self, model, ratio):
        # Calculate the importance score
        is_dict = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    is_dict[n] = p.abs().detach()
        # Calculate the mask threshold
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - ratio)))[0].item()
        # Mask weights whose importance lower than threshold
        current_mask = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    current_mask[n] = (is_dict[n] < mask_threshold)
                    p.masked_fill_(current_mask[n], self.masking_value)
        return mask_threshold, current_mask
   
    def magnitude_pruning(self, model, train_step_index):
        if train_step_index < self.cubic_prune_end:
            # Get the remaining ratio
            ratio, mask_ind = self.cubic_remaining_ratio_scheduler(train_step_index)
            if mask_ind and ratio < 1.0:
                # Mask weights during masking horizon
                mask_threshold, _ = self.mask_with_threshold(model, ratio)
            else:
                mask_threshold = None
        elif train_step_index == self.cubic_prune_end:
            ratio = self.final_ratio
            mask_threshold, final_ratio_mask = self.mask_with_threshold(model, ratio)
            self.final_ratio_mask = final_ratio_mask
        else:
            ratio = self.final_ratio
            mask_threshold = 0.0
            self.prune_masked_weights(model)
            
        return ratio, mask_threshold
    
    def cubic_remaining_ratio_scheduler(self, train_step_index):
        # Schedule the remaining ratio
        initial_ratio = self.initial_ratio
        final_ratio = self.final_ratio
        deltaT = self.deltaT
        deltaT_cooldown = self.deltaT_cooldown
        cubic_prune_start = self.cubic_prune_start
        cubic_prune_cool_down_start = self.cubic_prune_cool_down_start
        ratio_scheduler_steps = self.ratio_scheduler_steps
        mask_ind = False
        if train_step_index < cubic_prune_start:
            ratio = 1.0
            mask_ind = False
        elif train_step_index == cubic_prune_start:
            ratio = initial_ratio
            mask_ind = True
        elif train_step_index >= cubic_prune_cool_down_start:
            ratio = final_ratio
            mask_ind = True if train_step_index % deltaT_cooldown == 0 else False
        else:
            mul_coeff = 1 - (train_step_index - cubic_prune_start) / (ratio_scheduler_steps)
            ratio = final_ratio + (initial_ratio - final_ratio) * (mul_coeff ** 3)
            mask_ind = True if train_step_index % deltaT == 0 else False
        return ratio, mask_ind
    
    def prune_masked_weights(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    p.data.masked_fill_(self.final_ratio_mask[n], 0.0)

    def calculate_relative_sparsity(self, model):
        n_params = 0
        n_masked_params = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    n_params += p.numel()
                    n_masked_params += p.eq(0.0).sum().item()
        return n_masked_params/n_params
    
    def param_dist_stats(self, model):
        is_dict = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    is_dict[n] = p.abs().detach()
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        stats = {}
        quantiles = torch.tensor([0.25, 0.5, 0.75, 1.0]).to(all_is.device)
        q1, q2, q3, max = torch.quantile(all_is, quantiles, interpolation="nearest").tolist()
        stats["q1"] = q1
        stats["q2"] = q2
        stats["q3"] = q3
        stats["max"] = max
        return stats

    def print_pruning_modules(self, model):
        print ("list of model modules to be pruned:")
        for n, _ in model.named_parameters():
            if self.whether_mask_para(n):
                print (n)
        
    def zero_masked_para_grad(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    p.grad.masked_fill_(self.final_ratio_mask[n], 0.0)
    
    def gradient_clipping(self, model):
        self.zero_masked_para_grad(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clipping_threshold).item()
        return grad_norm

    def match(self, event, state):
        return event in [Event.FIT_START, Event.AFTER_TRAIN_BATCH, Event.BATCH_END, Event.FIT_END]

    def apply(self, event, state, logger):
        if event == Event.FIT_START:
            self.print_pruning_modules(state.model)
        elif event == Event.AFTER_TRAIN_BATCH:
            # add prior gradients to the model during the gradual cubic pruning stage
            if state.timestamp.batch.value <= self.cubic_prune_end:
                prior_threshold, sigma0, sigma1, lambda_mix = self.add_prior_grad(state.model, state.timestamp.batch.value)
                logger.log_metrics({"sigma0": float(sigma0)})
                logger.log_metrics({"sigma1": float(sigma1)})
                logger.log_metrics({"lambda_mix": float(lambda_mix)})
                logger.log_metrics({"prior_threshold": float(prior_threshold)})
            # perform gradient clipping during the optional sparse finetuning stage for non-masked parameters
            else:
                if self.clipping_threshold is not None:
                    grad_norm = self.gradient_clipping(state.model)
                    logger.log_metrics({"grad_norm": float(grad_norm)})
        elif event == Event.BATCH_END:
            ratio, mask_threshold = self.magnitude_pruning(state.model, state.timestamp.batch.value)
            logger.log_metrics({"remaining_ratio": float(ratio)})
            if mask_threshold is None:
                mask_threshold = 0.0
            logger.log_metrics({"mask_threshold": float(mask_threshold)})
            if self.log_param_stat_interval is not None and state.timestamp.batch.value % self.log_param_stat_interval == 0:
                stats = self.param_dist_stats(state.model)
                logger.log_metrics(stats)
        elif event == Event.FIT_END:
            relative_final_sparsity = self.calculate_relative_sparsity(state.model)
            logger.log_metrics({"relative_final_sparsity": float(relative_final_sparsity)})