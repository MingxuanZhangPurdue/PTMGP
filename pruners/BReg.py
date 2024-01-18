import torch
import re
import numpy as np
from composer.core import Algorithm, Event


def _get_unit_and_value(time):
    time_units = ["ep", "ba", "dur"]
    # regex for parsing time string, matches timeunit and chars prior to unit as value
    _TIME_STR_REGEX = re.compile(r'^(.+)(' + r'|'.join([fr'{time_unit}' for time_unit in time_units]) + r')$',
                                flags=re.IGNORECASE)
    match = _TIME_STR_REGEX.findall(time)
    if len(match) != 1:
        raise ValueError(f'Invalid time string: {time}')
    match = match[0]
    match = [x for x in match if x != '']
    assert len(match) == 2, 'each match should have a number followed by the key'
    value = match[0]
    unit = match[1]
    value = float(value)  # always parsing first as float b/c it could be scientific notation
    if unit == "ba":
        if int(value) != value:
            raise TypeError(f'value {value} is not an integer. Units {unit} require integer values.')
        value = int(value)
    return unit, value

def _convert_timestr_to_int(time, max_train_steps, train_dataloader_len):
    if isinstance(time, int):
        return time
    elif isinstance(time, str):
        unit, value = _get_unit_and_value(time)
        if unit.casefold() == "dur".casefold():
            return int(value*max_train_steps)
        elif unit.casefold() == "ba".casefold():
            return int(value)
        else:
            return int(value*train_dataloader_len)
    else:
        raise ValueError("time must be either int or str.")

class BReg(Algorithm):
    def __init__(self,
                 train_size, max_train_steps,
                 sigma0=1e-15, sigma1=0.1, lambda_mix=1e-3,
                 alpha_i_lambda=1.0, alpha_f_lambda=0.01,
                 anneal_start_lambda=None, anneal_end_lambda=None,
                 final_ratio=1, initial_ratio=1,
                 initial_warmup=0, final_warmup=0, deltaT=1,
                 masking_value=0.0,
                 non_mask_name=None,
                 non_prior_name=None):
                     
        self.train_size = train_size
        self.max_train_steps = max_train_steps
        self.masking_value = masking_value
        self.non_mask_name_pattern = re.compile("|".join(non_mask_name), re.IGNORECASE) if non_mask_name is not None else None
        self.non_prior_name_pattern = re.compile("|".join(non_prior_name), re.IGNORECASE) if non_prior_name is not None else None

        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.lambda_mix = lambda_mix
        self.alpha_i_lambda = alpha_i_lambda
        self.alpha_f_lambda = alpha_f_lambda

        self.final_ratio = final_ratio
        self.initial_ratio = initial_ratio
        self.initial_warmup = initial_warmup
        self.final_warmup = final_warmup
        self.deltaT = deltaT
        
        cubic_prune_start = initial_warmup
        cubic_prune_end = max_train_steps - final_warmup

        self.anneal_start_lambda = anneal_start_lambda if anneal_start_lambda is not None else cubic_prune_start
        self.anneal_end_lambda = anneal_end_lambda if anneal_end_lambda is not None else cubic_prune_end
        self.lambda_mix_scheduler_steps = self.anneal_end_lambda - self.anneal_start_lambda

        if not (cubic_prune_start <= cubic_prune_end <= max_train_steps):
            print ("cubic_prune_start:", cubic_prune_start)
            print ("cubic_prune_end:", cubic_prune_end)
            print ("max_train_steps:", max_train_steps)
            raise ValueError("cubic_prune_start <= cubic_prune_end <= max_train_steps is not satisfied.")
    
        self.ratio_scheduler_steps = cubic_prune_end - cubic_prune_start
        self.cubic_prune_start = cubic_prune_start
        self.cubic_prune_end = cubic_prune_end


    @classmethod
    def from_args(self, train_size, max_train_steps, train_dataloader_len, args):
        initial_warmup = _convert_timestr_to_int(args.initial_warmup, max_train_steps, train_dataloader_len)
        final_warmup = _convert_timestr_to_int(args.final_warmup, max_train_steps, train_dataloader_len)
        deltaT = _convert_timestr_to_int(args.deltaT, max_train_steps, train_dataloader_len)
        anneal_start_lambda = _convert_timestr_to_int(args.anneal_start_lambda, max_train_steps, train_dataloader_len) if args.anneal_start_lambda is not None else None
        anneal_end_lambda = _convert_timestr_to_int(args.anneal_end_lambda, max_train_steps, train_dataloader_len) if args.anneal_end_lambda is not None else None
        return self(train_size, max_train_steps,
                    sigma0=args.sigma0, sigma1=args.sigma1, lambda_mix=args.lambda_mix,
                    alpha_i_lambda=args.alpha_i_lambda, alpha_f_lambda=args.alpha_f_lambda,
                    anneal_start_lambda=anneal_start_lambda, anneal_end_lambda=anneal_end_lambda,
                    final_ratio=args.final_ratio, initial_ratio=args.initial_ratio,
                    initial_warmup=initial_warmup, final_warmup=final_warmup, deltaT=deltaT,
                    masking_value=args.masking_value, 
                    non_mask_name=args.non_mask_name, non_prior_name=args.non_prior_name
                    )

    def lambda_linear_scheduler(self, train_step_index):
        if train_step_index <= self.anneal_start_lambda:
            return self.alpha_i_lambda*self.lambda_mix
        else:
            if self.lambda_mix_scheduler_steps == 0 or self.alpha_i_lambda == self.alpha_f_lambda:
                return self.alpha_f_lambda*self.lambda_mix
            else:
                frac_of_total = min(1.0, (train_step_index - self.anneal_start_lambda) / (self.lambda_mix_scheduler_steps))
                current_factor = self.alpha_i_lambda + frac_of_total * (self.alpha_f_lambda - self.alpha_i_lambda)
                return current_factor*self.lambda_mix

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

    def calculate_prior_threshold(self, train_step_index):
        lambda_mix = self.lambda_linear_scheduler(train_step_index)
        sigma_1 = self.sigma1
        sigma_0 = self.sigma0
        c1 = np.log(lambda_mix) - np.log(1 - lambda_mix) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        prior_threshold = np.sqrt(np.log((1 - lambda_mix) / lambda_mix * np.sqrt(sigma_1 / sigma_0)) / (
                    0.5 / sigma_0 - 0.5 / sigma_1))
        return c1, c2, prior_threshold, lambda_mix
    
    def add_prior_grad(self, model, train_step_index):
        # Add prior gradients to the model       
        sigma_1 = self.sigma1
        sigma_0 = self.sigma0
        c1, c2, prior_threshold, lambda_mix = self.calculate_prior_threshold(train_step_index)
        if train_step_index <= self.cubic_prune_end:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if self.whether_penalize_para(n):
                        temp = p.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                        temp = temp.mul((sigma_0-sigma_1)/(self.train_size*sigma_0*sigma_1)).add((-1)/(self.train_size*sigma_1))
                        p.grad -= p.mul(temp)
        return prior_threshold, lambda_mix
    
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
            if mask_ind:
                # Mask weights during masking horizon
                mask_threshold, current_mask = self.mask_with_threshold(model, ratio)
            else:
                mask_threshold = None
        elif train_step_index == self.cubic_prune_end:
            ratio = self.final_ratio
            mask_threshold, current_mask = self.mask_with_threshold(model, ratio)
            self.mask = current_mask
        else:
            ratio = self.final_ratio
            mask_threshold = 0.0
            self.prune_masked_weights(model, self.mask)
            
        return ratio, mask_threshold
    
    def cubic_remaining_ratio_scheduler(self, train_step_index):
        # Schedule the remaining ratio
        initial_ratio = self.initial_ratio
        final_ratio = self.final_ratio
        deltaT = self.deltaT
        cubic_prune_start = self.cubic_prune_start
        cubic_prune_end = self.cubic_prune_end
        ratio_scheduler_steps = self.ratio_scheduler_steps
        mask_ind = False
        if train_step_index < cubic_prune_start:
            ratio = 1.0
            mask_ind = False
        elif train_step_index == cubic_prune_start:
            ratio = initial_ratio
            mask_ind = True
        elif train_step_index >= cubic_prune_end:
            ratio = final_ratio
            mask_ind = True
        else:
            mul_coeff = 1 - (train_step_index - cubic_prune_start) / (ratio_scheduler_steps)
            ratio = final_ratio + (initial_ratio - final_ratio) * (mul_coeff ** 3)
            mask_ind = True if train_step_index % deltaT == 0 else False
        return ratio, mask_ind
    
    def prune_masked_weights(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    p.data.masked_fill_(self.mask[n], 0.0)

    def calculate_relative_sparsity(self, model):
        n_params = 0
        n_masked_params = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    n_params += p.numel()
                    n_masked_params += p.eq(0.0).sum().item()
        return n_masked_params/n_params

    def match(self, event, state):
        return event in [Event.AFTER_TRAIN_BATCH,  Event.BATCH_END, Event.FIT_END]

    def apply(self, event, state, logger):
        if event == Event.AFTER_TRAIN_BATCH:
            prior_threshold, lambda_mix = self.add_prior_grad(state.model, state.timestamp.batch.value)
            logger.log_metrics({"lambda_mix": float(lambda_mix)})
            logger.log_metrics({"prior_threshold": float(prior_threshold)})
        elif event == Event.BATCH_END:
            ratio, mask_threshold = self.magnitude_pruning(state.model, state.timestamp.batch.value)
            logger.log_metrics({"remaining_ratio": float(ratio)})
            if mask_threshold is None:
                mask_threshold = 0.0
            logger.log_metrics({"mask_threshold": float(mask_threshold)})
        elif event == Event.FIT_END:
            relative_final_sparsity = self.calculate_relative_sparsity(state.model)
            logger.log_metrics({"relative_final_sparsity": float(relative_final_sparsity)})