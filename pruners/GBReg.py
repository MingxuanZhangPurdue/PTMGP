import torch
import re
import wandb
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
    
def _calculate_n_reselection(mask1, mask2):
    n_same = 0
    n = 0
    for key in mask1.keys():
        n += mask1[key].numel()
        n_same += torch.eq(mask1[key], mask2[key]).sum().item()
    n_diff = n - n_same
    return n_diff

class GBReg(Algorithm):
    def __init__(
            self,
            # total number of training samples
            train_size,
            # total number of training steps
            max_train_steps,
            # sigma0 annealing configuration
            sigma0=1e-15, # initial value of sigma0
            alpha_i_sigma0=1.0, # initial value of alpha (factor) for sigma0
            alpha_f_sigma0=1.0, # final value of alpha (factor) for sigma0
            anneal_start_sigma0=None, # start step of annealing for sigma0
            anneal_end_sigma0=None, # end step of annealing for sigma0
            # sigma1 annealing configuration
            sigma1=0.05, # initial value of sigma1
            alpha_i_sigma1=1.0, # initial value of alpha (factor) for sigma1
            alpha_f_sigma1=1.0, # final value of alpha (factor) for sigma1
            anneal_start_sigma1=None, # start step of annealing for sigma1
            anneal_end_sigma1=None, # end step of annealing for sigma1
            # lambda_mix annealing configuration
            lambda_mix=1e-3, # initial value of lambda_mix
            alpha_i_lambda_mix=1.0, # initial value of alpha (factor) for lambda_mix
            alpha_f_lambda_mix=1.0, # final value of alpha (factor) for lambda_mix
            anneal_start_lambda_mix=None, # start step of annealing for lambda_mix
            anneal_end_lambda_mix=None, # end step of annealing for lambda_mix
            # initial sparsity ratio, if set to less than 1.0, will prune the model to this ratio at the beginning of the cubic graudal pruning stage
            initial_ratio=1.0,
            # target remaining ratio, i.e., final_ratio = 1 - the target sparsity
            final_ratio=0.1,
            # number of steps for the initial warmup stage
            initial_warmup_steps=0,
            # number of steps for the final warmup stage
            final_warmup_steps=0,
            # masking horizon for the gradual cubic pruning stage
            deltaT=10,
            # masking horizon for the final warmup stage
            deltaT_final_warmup=1,
            # what degree of prior regularization to use during the final warmup stage
            # annealed: use the annealed prior regularization
            # initial:  use the initial prior regularization
            # none:     no prior regularization
            final_warmup_prior_config="none",
            # masking value for the pruned weights
            masking_value=0.0,
            # parmeter names that should not be masked, matched by regular expression
            non_mask_name=None, 
            # parmeter names that should not be penalized, matched by regular expression
            non_prior_name=None,
            # gradient norm clipping threshold, used when no prior imposed during the final warmup stage
            clipping_threshold=1.0,
            # log parameter's magnitude statistics
            param_magnitude_stat_log_interval=None,
            # log how mask corresponds to the final ratio changes during the gradual cubic pruning stage
            mask_update_log_interval=None,
            # log the number of parameters in the high penalty region, i.e., the spike region
            log_spike_region = False,
        ):

        self.final_ratio_mask_after_initial_warmup = None
        self.current_mask = None

        self.train_size = train_size
        self.max_train_steps = max_train_steps

        self.param_magnitude_stat_log_interval = param_magnitude_stat_log_interval
        self.mask_update_log_interval = mask_update_log_interval
        self.log_spike_region = log_spike_region

        self.clipping_threshold = clipping_threshold

        self.masking_value = masking_value
        self.non_mask_name_pattern = re.compile("|".join(non_mask_name), re.IGNORECASE) if non_mask_name is not None else None
        self.non_prior_name_pattern = re.compile("|".join(non_prior_name), re.IGNORECASE) if non_prior_name is not None else None

        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.deltaT = deltaT
        self.deltaT_final_warmup = deltaT_final_warmup
        self.final_warmup_prior_config = final_warmup_prior_config

        cubic_prune_start = initial_warmup_steps
        cubic_prune_end = max_train_steps - final_warmup_steps

        self.sigma0 = sigma0
        self.alpha_i_sigma0 = alpha_i_sigma0
        self.alpha_f_sigma0 = alpha_f_sigma0
        self.anneal_start_sigma0 = anneal_start_sigma0 if anneal_start_sigma0 is not None else cubic_prune_start
        self.anneal_end_sigma0 = anneal_end_sigma0 if anneal_end_sigma0 is not None else cubic_prune_end

        self.sigma1 = sigma1
        self.alpha_i_sigma1 = alpha_i_sigma1
        self.alpha_f_sigma1 = alpha_f_sigma1
        self.anneal_start_sigma1 = anneal_start_sigma1 if anneal_start_sigma1 is not None else cubic_prune_start
        self.anneal_end_sigma1 = anneal_end_sigma1 if anneal_end_sigma1 is not None else cubic_prune_end

        self.lambda_mix = lambda_mix
        self.alpha_i_lambda_mix = alpha_i_lambda_mix
        self.alpha_f_lambda_mix = alpha_f_lambda_mix
        self.anneal_start_lambda_mix = anneal_start_lambda_mix if anneal_start_lambda_mix is not None else cubic_prune_start
        self.anneal_end_lambda_mix = anneal_end_lambda_mix if anneal_end_lambda_mix is not None else cubic_prune_end

        if not (cubic_prune_start < cubic_prune_end <= max_train_steps):
            print ("cubic_prune_start:", cubic_prune_start)
            print ("cubic_prune_end:", cubic_prune_end)
            print ("max_train_steps:", max_train_steps)
            raise ValueError("cubic_prune_start < cubic_prune_end <= max_train_steps must be satisfied, but got False")
    
        self.ratio_scheduler_steps = cubic_prune_end - cubic_prune_start
        self.cubic_prune_start = cubic_prune_start
        self.cubic_prune_end = cubic_prune_end

    @classmethod
    def from_args(self, train_size, max_train_steps, train_dataloader_len, args):
        initial_warmup_steps = _convert_timestr_to_int(args.initial_warmup_steps, max_train_steps, train_dataloader_len)
        final_warmup_steps = _convert_timestr_to_int(args.final_warmup_steps, max_train_steps, train_dataloader_len)
        deltaT = _convert_timestr_to_int(args.deltaT, max_train_steps, train_dataloader_len)
        deltaT_final_warmup = _convert_timestr_to_int(args.deltaT_final_warmup, max_train_steps, train_dataloader_len)
        anneal_start_sigma0 = _convert_timestr_to_int(args.anneal_start_sigma0, max_train_steps, train_dataloader_len) if args.anneal_start_sigma0 is not None else None
        anneal_end_sigma0 = _convert_timestr_to_int(args.anneal_end_sigma0, max_train_steps, train_dataloader_len) if args.anneal_end_sigma0 is not None else None
        anneal_start_sigma1 = _convert_timestr_to_int(args.anneal_start_sigma1, max_train_steps, train_dataloader_len) if args.anneal_start_sigma1 is not None else None
        anneal_end_sigma1 = _convert_timestr_to_int(args.anneal_end_sigma1, max_train_steps, train_dataloader_len) if args.anneal_end_sigma1 is not None else None
        anneal_start_lambda_mix = _convert_timestr_to_int(args.anneal_start_lambda_mix, max_train_steps, train_dataloader_len) if args.anneal_start_lambda_mix is not None else None
        anneal_end_lambda_mix = _convert_timestr_to_int(args.anneal_end_lambda_mix, max_train_steps, train_dataloader_len) if args.anneal_end_lambda_mix is not None else None
        param_magnitude_stat_log_interval = _convert_timestr_to_int(args.param_magnitude_stat_log_interval, max_train_steps, train_dataloader_len) if args.param_magnitude_stat_log_interval is not None else None
        mask_update_log_interval = _convert_timestr_to_int(args.mask_update_log_interval, max_train_steps, train_dataloader_len) if args.mask_update_log_interval is not None else None
        return self(
            train_size=train_size,
            max_train_steps=max_train_steps,
            sigma0=args.sigma0,
            alpha_i_sigma0=args.alpha_i_sigma0,
            alpha_f_sigma0=args.alpha_f_sigma0,
            anneal_start_sigma0=anneal_start_sigma0,
            anneal_end_sigma0=anneal_end_sigma0,
            sigma1=args.sigma1,
            alpha_i_sigma1=args.alpha_i_sigma1,
            alpha_f_sigma1=args.alpha_f_sigma1,
            anneal_start_sigma1=anneal_start_sigma1,
            anneal_end_sigma1=anneal_end_sigma1,
            lambda_mix=args.lambda_mix,
            alpha_i_lambda_mix=args.alpha_i_lambda_mix,
            alpha_f_lambda_mix=args.alpha_f_lambda_mix,
            anneal_start_lambda_mix=anneal_start_lambda_mix,
            anneal_end_lambda_mix=anneal_end_lambda_mix,
            initial_ratio=args.initial_ratio,
            final_ratio=args.final_ratio,
            initial_warmup_steps=initial_warmup_steps,
            final_warmup_steps=final_warmup_steps,
            deltaT=deltaT,
            deltaT_final_warmup=deltaT_final_warmup,
            final_warmup_prior_config=args.final_warmup_prior_config,
            masking_value=args.masking_value,
            non_mask_name=args.non_mask_name,
            non_prior_name=args.non_prior_name,
            clipping_threshold=args.clipping_threshold,
            param_magnitude_stat_log_interval=param_magnitude_stat_log_interval,
            mask_update_log_interval=mask_update_log_interval,
            log_spike_region=args.log_spike_region
        )

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
        
    def linear_prior_scheduler(self, train_step_index):
        if train_step_index > self.cubic_prune_end and self.final_warmup_prior_config == "initial":
            lambda_mix_factor = self.alpha_i_lambda_mix
            sigma0_factor = self.alpha_i_sigma0
            sigma1_factor = self.alpha_i_sigma1
        else:
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
            sigma1_factor = _linear_scheduler(
                train_step_index, 
                self.anneal_start_sigma1, 
                self.anneal_end_sigma1, 
                self.alpha_i_sigma1, 
                self.alpha_f_sigma1
            )
        return sigma0_factor*self.sigma0, sigma1_factor*self.sigma1, lambda_mix_factor*self.lambda_mix

    def calculate_prior_grad_components(self, train_step_index):
        sigma0, sigma1, lambda_mix = self.linear_prior_scheduler(train_step_index)
        c1 = np.log(lambda_mix) - np.log(1 - lambda_mix) + 0.5 * np.log(sigma0) - 0.5 * np.log(sigma1)
        c2 = 0.5 / sigma0 - 0.5 / sigma1
        prior_threshold = np.sqrt(np.log((1 - lambda_mix) / lambda_mix * np.sqrt(sigma1 / sigma0)) / (
                    0.5 / sigma0 - 0.5 / sigma1))
        return c1, c2, prior_threshold, sigma0, sigma1, lambda_mix
    
    def add_prior_grad(self, model, train_step_index):
        # add prior gradients to the model during the gradual cubic pruning stage,
        c1, c2, prior_threshold, sigma0, sigma1, lambda_mix = self.calculate_prior_grad_components(train_step_index)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_penalize_para(n):
                    temp = p.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                    temp = temp.mul((sigma0-sigma1)/(self.train_size*sigma0*sigma1)).add((-1)/(self.train_size*sigma1))
                    p.grad -= p.mul(temp)
        return prior_threshold, sigma0, sigma1, lambda_mix
    
    def calculate_mask_threshold(self, model, ratio):
        # Calculate the importance score
        is_dict = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    is_dict[n] = p.abs().detach()
        # Calculate the mask threshold
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - ratio)))[0].item()
        return mask_threshold, is_dict
    
    def create_mask(self, model, mask_threshold, is_dict):
        # Create mask
        current_mask = {}
        for n, _ in model.named_parameters():
            if self.whether_mask_para(n):
                current_mask[n] = (is_dict[n] < mask_threshold)
        return current_mask
    
    def mask_with_threshold(self, model, ratio):
        mask_threshold, is_dict = self.calculate_mask_threshold(model, ratio)
        current_mask = self.create_mask(model, mask_threshold, is_dict)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    p.masked_fill_(current_mask[n], self.masking_value)
        return mask_threshold, current_mask
   
    def magnitude_pruning(self, model, train_step_index):
        ratio, mask_ind = self.cubic_remaining_ratio_scheduler(train_step_index)
        if mask_ind and ratio < 1.0:
                # Mask weights during masking horizon
                mask_threshold, mask = self.mask_with_threshold(model, ratio)
        else:
            mask_threshold = None
            mask = None
        return ratio, mask_threshold, mask
    
    def cubic_remaining_ratio_scheduler(self, train_step_index):
        # schedule the remaining ratio
        initial_ratio = self.initial_ratio
        final_ratio = self.final_ratio
        deltaT = self.deltaT
        deltaT_final_warmup = self.deltaT_final_warmup
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
        elif train_step_index == cubic_prune_end:
            ratio = final_ratio
            mask_ind = True
        elif train_step_index > cubic_prune_end:
            ratio = final_ratio
            mask_ind = True if train_step_index % deltaT_final_warmup == 0 else False
        else:
            mul_coeff = 1 - (train_step_index - cubic_prune_start) / (ratio_scheduler_steps)
            ratio = final_ratio + (initial_ratio - final_ratio) * (mul_coeff ** 3)
            mask_ind = True if train_step_index % deltaT == 0 else False
        return ratio, mask_ind
    
    def gradient_clipping(self, model):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clipping_threshold).item()
        return grad_norm

    def calculate_relative_sparsity(self, model):
        n_params = 0
        n_masked_params = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    n_params += p.numel()
                    n_masked_params += p.eq(0.0).sum().item()
        return n_masked_params/n_params
    
    def magnitude_stat(self, model, mask=None):
        magnitude_vector = []
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    if mask is None:
                        magnitude_vector.append(p.abs().detach().view(-1))
                    else:
                        magnitude_vector.append(p.abs().detach()[~mask[n]].view(-1))
        magnitude_vector = torch.cat(magnitude_vector)
        magnitude_stat = {}
        magnitude_stat["avg"] = float(magnitude_vector.mean())
        magnitude_stat["std"] = float(magnitude_vector.std())
        return magnitude_stat
    
    def calculate_n_param_below_prior_threshold(self, model, prior_threshold):
        n_param = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    n_param += (p.abs() <= prior_threshold).sum().item()
        return n_param
    
    def print_pruning_modules(self, model):
        print ("list of model modules to be pruned:")
        for n, _ in model.named_parameters():
            if self.whether_mask_para(n):
                print (n)

    def match(self, event, state):
        return event in [Event.FIT_START, Event.AFTER_TRAIN_BATCH, Event.BATCH_END]

    def apply(self, event, state, logger):
        if event == Event.FIT_START:
            # print the list of model modules to be pruned
            self.print_pruning_modules(state.model)
            # log the parameter's magnitude statistics of the pre-trained model
            if self.param_magnitude_stat_log_interval is not None:
                magnitude_stat = self.magnitude_stat(state.model)
                logger.log_metrics(magnitude_stat)
        elif event == Event.AFTER_TRAIN_BATCH:
            # add prior gradients to the model during the gradual cubic pruning stage
            if state.timestamp.batch.value <= self.cubic_prune_end or self.final_warmup_prior_config != "none":
                prior_threshold, sigma0, sigma1, lambda_mix = self.add_prior_grad(state.model, state.timestamp.batch.value)
                logger.log_metrics({"sigma0": float(sigma0)})
                logger.log_metrics({"sigma1": float(sigma1)})
                logger.log_metrics({"lambda_mix": float(lambda_mix)})
                logger.log_metrics({"prior_threshold": float(prior_threshold)})
                # log the number of parameters in the high penalty region, i.e., the spike region
                if self.log_spike_region and state.timestamp.batch.value < self.cubic_prune_end and (state.timestamp.batch.value-1) % self.deltaT == 0:
                    n_param_below_prior_threshold = self.calculate_n_param_below_prior_threshold(state.model, prior_threshold)
                    logger.log_metrics({"n_param_below_prior_threshold": int(n_param_below_prior_threshold)})
            # perform gradient clipping during the final warmup stage if no prior regularization is imposed
            if state.timestamp.batch.value > self.cubic_prune_end and self.final_warmup_prior_config == "none" and self.clipping_threshold is not None:
                grad_norm = self.gradient_clipping(state.model)
                logger.log_metrics({"grad_norm": float(grad_norm)})
        elif event == Event.BATCH_END:
            # log how mask corresponds to the final ratio changes during the gradual cubic pruning stage
            if self.mask_update_log_interval is not None:
                if state.timestamp.batch.value == self.cubic_prune_start:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                    self.final_ratio_mask_after_initial_warmup = self.create_mask(state.model, mask_threshold, is_dict)
                if state.timestamp.batch.value % self.mask_update_log_interval == 0:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                    updated_final_ratio_mask = self.create_mask(state.model, mask_threshold, is_dict)
                    n_reselection = _calculate_n_reselection(self.final_mask_after_initial_warmup, updated_final_ratio_mask)
                    logger.log_metrics({"n_reselection": int(n_reselection)})
            # perform magnitude pruning
            ratio, mask_threshold, mask = self.magnitude_pruning(state.model, state.timestamp.batch.value)
            # log the current remaining ratio
            logger.log_metrics({"remaining_ratio": float(ratio)})
            # if the current mask is not None, update the current mask
            if mask is not None:
                self.current_mask = mask
            # if the current mask threshold is not None, log the current mask threshold
            if mask_threshold is not None:
                logger.log_metrics({"mask_threshold": float(mask_threshold)})
            # log the parameter's magnitude statistics
            if self.param_magnitude_stat_log_interval is not None and state.timestamp.batch.value % self.param_magnitude_stat_log_interval == 0:
                if state.timestamp.batch.value < self.cubic_prune_start:
                    magnitude_stat = self.magnitude_stat(state.model)
                else:
                    magnitude_stat = self.magnitude_stat(state.model, self.current_mask)
                logger.log_metrics(magnitude_stat)
