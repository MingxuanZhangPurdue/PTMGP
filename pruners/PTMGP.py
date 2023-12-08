import torch
import numpy as np


def set_PMGP(model, train_size, max_train_steps, args):

    return 0




class PMGP(object):
    
    def __init__(self, model, train_size, max_train_steps,
                 final_ratio, initial_ratio=1,
                 sigma0=1e-12, sigma1=0.1, lambda_mix=1e-7,
                 anneal_start = 0, anneal_end = 0,
                 initial_warmup = 0.0, final_warmup = 1, warmup_steps = 1000,
                 non_mask_name=None):
        
        self.train_size = train_size
        self.model = model
        self.max_train_steps = max_train_steps
        self.non_mask_name = non_mask_name

        self.lambda_mix = lambda_mix
        self.sigma0 = sigma0
        self.sigma1 = sigma1

        self.anneal_start = anneal_start
        self.anneal_end = anneal_end

        self.final_ratio = final_ratio
        self.initial_ratio = initial_ratio
        
        self.initial_warmup = initial_warmup
        self.final_warmup = final_warmup
        self.warmup_steps = warmup_steps
        
        cubic_prune_start = anneal_end + initial_warmup*warmup_steps
        cubic_prune_end = max_train_steps - final_warmup*warmup_steps
        
        if not (anneal_end <= cubic_prune_start <= cubic_prune_end <= max_train_steps):
            print ("anneal_end:", anneal_end)
            print ("cubic_prune_start:", cubic_prune_start)
            print ("cubic_prune_end:", cubic_prune_end)
            print ("max_train_steps:", max_train_steps)
            raise ValueError("Wrong schedule values!")
    
        self.spars_schedu_steps = cubic_prune_end - cubic_prune_start
        self.cubic_prune_start = cubic_prune_start
        self.cubic_prune_end = cubic_prune_end

    
    def calculate_prior_threshold(self):

        lambda_mix = self.lambda_mix
        sigma_1 = self.sigma1
        sigma_0 = self.sigma0


        c1 = np.log(lambda_mix) - np.log(1 - lambda_mix) + 0.5 * np.log(sigma_0) - 0.5 * np.log(sigma_1)
        c2 = 0.5 / sigma_0 - 0.5 / sigma_1
        prior_threshold = np.sqrt(np.log((1 - lambda_mix) / lambda_mix * np.sqrt(sigma_1 / sigma_0)) / (
                    0.5 / sigma_0 - 0.5 / sigma_1))
        
        return c1, c2, prior_threshold
    
    def add_prior_grad(self, train_step_index):
        
        anneal_start = self.prior_start
        anneal_end = self.anneal_end
        prior_warmup_steps = self.prior_warmup_steps
        
        if train_step_index < anneal_start:
            anneal_lambda = 0
        elif anneal_start <= train_step_index < anneal_end:
            anneal_lambda = 1.0 * (train_step_index - anneal_start)/prior_warmup_steps
        else:
            anneal_lambda = 1.0
                
        lambda_n = self.config["lambdan"]
        prior_sigma_1 = self.config["sigma1"]
        prior_sigma_0 = self.config["sigma0"]
        c1, c2, prior_threshold = self.calculate_prior_threshold()
        
        if anneal_lambda > 0:
            with torch.no_grad():
                for pname, param in self.model.named_parameters():
                    if self.non_mask_name == None or not any(nd in pname for nd in self.non_mask_name):
                        temp = param.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                        temp = param.div(-prior_sigma_0).mul(temp) + param.div(-prior_sigma_1).mul(1 - temp)
                        prior_grad = temp.div(self.train_size)
                        param.grad.data -= anneal_lambda*prior_grad
                    
        return prior_threshold
    
    # some functions below are adopted and modified based on https://github.com/QingruZhang/PLATON/blob/main/Pruner.py
    def mask_with_threshold(self, model, threshold):
        
        # Calculate the final importance score
        is_dict = {}
        for n, p in model.named_parameters():
            if self.non_mask_name == None or not any(nd in n for nd in self.non_mask_name):
                is_dict[n] = p.abs().detach()
                    
        # Calculate the mask threshold
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * (1 - threshold)))[0].item()
        
        # Mask weights whose importance lower than threshold
        for n, p in model.named_parameters():
            if self.non_mask_name == None or not any(nd in n for nd in self.non_mask_name):
                p.data.masked_fill_(is_dict[n] < mask_threshold, 0.0)
                
        return mask_threshold
   
    def magnitude_pruning(self, model, train_step_index):
        # Get the remaining ratio
        threshold, mask_ind = self.schedule_threshold_comb(train_step_index)
        if mask_ind:
            # Mask weights during masking horizon
            mask_threshold = self.mask_with_threshold(model, threshold)
        else:
            mask_threshold = None
        
        return threshold, mask_threshold
    
    def schedule_threshold_comb(self, step):
        
        # Schedule the remaining ratio
        initial_threshold = self.config['initial_threshold']
        final_threshold = self.config['final_threshold']
        deltaT = self.config['deltaT']
        cubic_prune_start = self.config["cubic_prune_start"]
        cubic_prune_end = self.config["cubic_prune_end"]
        spars_schedu_steps = self.config["spars_schedu_steps"]
        mask_ind = False
        if step <= cubic_prune_start:
            threshold = initial_threshold
            mask_ind = False
        elif step > cubic_prune_end:
            threshold = final_threshold
            mask_ind = True
        else:
            mul_coeff = 1 - (step - cubic_prune_start) / (spars_schedu_steps)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
            mask_ind = True if step % deltaT == 0 else False
        return threshold, mask_ind