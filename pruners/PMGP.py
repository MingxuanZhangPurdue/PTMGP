import torch
import re
import numpy as np
from composer.core import Algorithm, Event

class PMGP_Algorithm(Algorithm):
    def __init__(self, 
                 train_size, max_train_steps,
                 final_ratio=1, initial_ratio=1,
                 sigma0=1e-12, sigma1=0.1, lambda_mix=1e-7,
                 anneal_start = 0, anneal_end = 0,
                 initial_warmup = 0.0, final_warmup = 1, deltaT = 1,
                 masking_value = 0.0, apply_prior_on_all_layers=False, non_mask_name=None):
        
        self.train_size = train_size
        self.max_train_steps = max_train_steps
        self.non_mask_name_pattern = re.compile("|".join(non_mask_name), re.IGNORECASE) if non_mask_name is not None else None
        self.masking_value = masking_value
        self.apply_prior_on_all_layers = apply_prior_on_all_layers

        self.lambda_mix = lambda_mix
        self.sigma0 = sigma0
        self.sigma1 = sigma1

        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        self.prior_warmup = anneal_end - anneal_start

        self.final_ratio = final_ratio
        self.initial_ratio = initial_ratio
        
        self.initial_warmup = initial_warmup
        self.final_warmup = final_warmup
        self.deltaT = deltaT
        
        cubic_prune_start = anneal_end + initial_warmup
        cubic_prune_end = max_train_steps - final_warmup
        
        if not (anneal_end <= cubic_prune_start <= cubic_prune_end <= max_train_steps):
            print ("anneal_end:", anneal_end)
            print ("cubic_prune_start:", cubic_prune_start)
            print ("cubic_prune_end:", cubic_prune_end)
            print ("max_train_steps:", max_train_steps)
            raise ValueError("Wrong scheduler values!")
    
        self.ratio_scheduler_steps = cubic_prune_end - cubic_prune_start
        self.cubic_prune_start = cubic_prune_start
        self.cubic_prune_end = cubic_prune_end

    @classmethod
    def from_args(self, train_size, max_train_steps, args):
        return self(train_size, max_train_steps,
                    final_ratio=args.final_ratio, initial_ratio=args.initial_ratio,
                    sigma0=args.sigma0, sigma1=args.sigma1, lambda_mix=args.lambda_mix,
                    anneal_start=args.anneal_start, anneal_end=args.anneal_end,
                    initial_warmup=args.initial_warmup, final_warmup=args.final_warmup,
                    deltaT=args.deltaT,
                    masking_value=args.masking_value, 
                    apply_prior_on_all_layers=args.apply_prior_on_all_layers, 
                    non_mask_name=args.non_mask_name,
                    )

    def whether_mask_para(self, n):
        if self.non_mask_name_pattern == None:
            return True
        else:
            return not bool(re.search(self.non_mask_name_pattern, n))

    def calculate_prior_threshold(self):

        lambda_mix = self.lambda_mix
        sigma_1 = self.sigma1
        sigma_0 = self.sigma0


        c1 = np.log(lambda_mix) - np.log(1 - lambda_mix) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        prior_threshold = np.sqrt(np.log((1 - lambda_mix) / lambda_mix * np.sqrt(sigma_1 / sigma_0)) / (
                    0.5 / sigma_0 - 0.5 / sigma_1))
        
        return c1, c2, prior_threshold
    
    def add_prior_grad(self, model, train_step_index):
        
        anneal_start = self.anneal_start
        anneal_end = self.anneal_end
        prior_warmup = self.prior_warmup
        
        if train_step_index < anneal_start:
            anneal_lambda = 0
        elif anneal_start <= train_step_index < anneal_end:
            anneal_lambda = 1.0 * (train_step_index - anneal_start)/prior_warmup
        else:
            anneal_lambda = 1.0
                
        sigma_1 = self.sigma1
        sigma_0 = self.sigma0
        c1, c2, prior_threshold = self.calculate_prior_threshold()
        
        if anneal_lambda > 0:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if self.apply_prior_on_all_layers or self.whether_mask_para(n):
                        temp = p.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                        temp = p.div(-sigma_0).mul(temp) + p.div(-sigma_1).mul(1 - temp)
                        prior_grad = temp.div(self.train_size)
                        p.grad.data -= anneal_lambda*prior_grad
        return prior_threshold, anneal_lambda
    
    def mask_with_threshold(self, model, ratio):
        
        # Calculate the importance score
        is_dict = {}
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                is_dict[n] = p.abs().detach()
                    
        # Calculate the mask threshold
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - ratio)))[0].item()
        
        # Mask weights whose importance lower than threshold
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                p.data.masked_fill_(is_dict[n] < mask_threshold, self.masking_value)
                
        return mask_threshold
   
    def magnitude_pruning(self, model, train_step_index):
        
        # Get the remaining ratio
        ratio, mask_ind = self.cubic_remaining_ratio_scheduler(train_step_index)
        if mask_ind:
            # Mask weights during masking horizon
            mask_threshold = self.mask_with_threshold(model, ratio)
        else:
            mask_threshold = None
        
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
        if train_step_index <= cubic_prune_start:
            ratio = initial_ratio
            mask_ind = False
        elif train_step_index > cubic_prune_end:
            ratio = final_ratio
            mask_ind = True
        else:
            mul_coeff = 1 - (train_step_index - cubic_prune_start) / (ratio_scheduler_steps)
            ratio = final_ratio + (initial_ratio - final_ratio) * (mul_coeff ** 3)
            mask_ind = True if train_step_index % deltaT == 0 else False
        return ratio, mask_ind
    
    @torch.no_grad()
    def calculate_sparsity(self, model):
        n_params = 0
        n_masked_params = 0
        for n, p in model.named_parameters():
            if self.whether_mask_para(n):
                n_params += p.numel()
                n_masked_params += p.data.eq(0.0).sum().item()
        return n_masked_params/n_params

    def match(self, event, state):
        return event in [Event.AFTER_TRAIN_BATCH,  Event.BATCH_END, Event.FIT_END]

    def apply(self, event, state, logger):
        if event == Event.AFTER_TRAIN_BATCH:
            self.add_prior_grad(state.model, state.timestamp.batch.value)
        elif event == Event.BATCH_END:
            ratio, mask_threshold = self.magnitude_pruning(state.model, state.timestamp.batch.value)
            logger.log_metrics({"remaining_ratio": float(ratio)})
            if mask_threshold is None:
                mask_threshold = 0.0
            logger.log_metrics({"mask_threshold": float(mask_threshold)})
        elif event == Event.FIT_END:
            final_sparsity = self.calculate_sparsity(state.model)
            logger.log_metrics({"final_sparsity": float(final_sparsity)})