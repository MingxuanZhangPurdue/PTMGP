import torch
import re
from composer.core import Algorithm, Event

class PLATON_Algorithm(Algorithm):
    def __init__(self, 
                 max_train_steps, 
                 beta1=0.85, beta2=0.95,
                 initial_ratio=1, final_ratio=0.5,
                 initial_warmup=1, final_warmup=1, warmup_steps=1, deltaT=1,
                 non_mask_name=None):

        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

        self.max_train_steps = max_train_steps

        self.beta1 = beta1
        self.beta2 = beta2

        self.final_ratio = final_ratio
        self.initial_ratio = initial_ratio
        self.initial_warmup = initial_warmup
        self.final_warmup = final_warmup
        self.warmup_steps = warmup_steps
        self.deltaT = deltaT

        self.non_mask_name_pattern = re.compile("|".join(non_mask_name), re.IGNORECASE) if non_mask_name is not None else None

        cubic_prune_start = initial_warmup*warmup_steps
        cubic_prune_end = max_train_steps - final_warmup*warmup_steps

        self.cubic_prune_start = cubic_prune_start
        self.cubic_prune_end = cubic_prune_end
        self.ratio_scheduler_steps = cubic_prune_end - cubic_prune_start
        
    @classmethod
    def from_args(self, max_train_steps, args):

        return self(max_train_steps=max_train_steps, 
                    beta1=args.beta1, beta2=args.beta2,
                    initial_ratio=args.initial_ratio, final_ratio=args.final_ratio,
                    initial_warmup=args.initial_warmup, final_warmup=args.final_warmup,
                    warmup_steps=args.warmup_steps, deltaT=args.deltaT,
                    non_mask_name=args.non_mask_name
                    )
    
    def whether_mask_para(self, n):

        if self.non_mask_name_pattern == None:
            return True
        else:
            return not bool(re.search(self.non_mask_name_pattern, n))

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


    def update_ipt_with_local_window(self, model, global_step):

        # Calculate the sensitivity and uncertainty 
        for n,p in model.named_parameters():
            if self.whether_mask_para(n):
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.ipt[n] = torch.zeros_like(p)
                    if self.beta2>0 and self.beta2!=1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                
                local_step = global_step % self.deltaT
                update_step = global_step // self.deltaT
                if local_step == 0: 
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    if self.beta2 > 0 and self.beta2 < 1:
                        self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                                (1 - self.beta2) * (self.ipt[n]-self.exp_avg_ipt[n]).abs()
                    elif self.beta2 == 2.:
                        self.exp_avg_unc[n] = (update_step*self.exp_avg_unc[n] + \
                                                (self.ipt[n]-self.exp_avg_ipt[n])**2 )/(update_step+1)
                    self.ipt[n] = (p * p.grad).abs().detach()
                else:
                    self.ipt[n] = (self.ipt[n]*local_step+(p*p.grad).abs().detach())/(local_step+1)


    def mask_with_threshold(self, model, ratio):

        # Calculate the final importance score
        is_dict = {}
        for n,p in model.named_parameters():
            if self.whether_mask_para(n):
                if self.beta2 > 0 and self.beta2<1:
                    is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n] 
                elif self.beta2 == 1.:
                    is_dict[n] = self.exp_avg_ipt[n]
                elif self.beta2 == 2.:
                    is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc.sqrt()
                else:
                    # Handling the uncepted beta2 to default setting 
                    is_dict[n] = self.exp_avg_ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
        # Calculate the mask threshold 
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0]*(1 - ratio)))[0].item()
        # Mask weights whose importance lower than threshold 
        for n,p in model.named_parameters():
            if self.whether_mask_para(n):
                p.data.masked_fill_(is_dict[n] < mask_threshold, 0.0)
        return mask_threshold


    def update_and_pruning(self, model, train_step_index):

        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, train_step_index)
        # Get the remaining ratio
        ratio, mask_ind = self.cubic_remaining_ratio_scheduler(train_step_index)
        if mask_ind:
            # Mask weights during masking horizon
            mask_threshold = self.mask_with_threshold(model, ratio)
        else:
            mask_threshold = None
        
        return ratio, mask_threshold
    
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

        return event in [Event.BATCH_END, Event.FIT_END]

    def apply(self, event, state, logger):
        
        if event == Event.BATCH_END:
            ratio, mask_threshold = self.update_and_pruning(state.model, state.timestamp.batch.value)
            logger.log_metrics({"remaining_ratio": float(ratio)})
            if mask_threshold is None:
                mask_threshold = 0.0
            logger.log_metrics({"mask_threshold": float(mask_threshold)})
        elif event == Event.FIT_END:
            final_sparsity = self.calculate_sparsity(state.model)
            logger.log_metrics({"final_sparsity": float(final_sparsity)})