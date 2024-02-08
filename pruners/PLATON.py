import torch
import re
from composer.core import Algorithm, Event
from pruners.utils_composer import _convert_timestr_to_int

def _calculate_n_reselection(mask1, mask2):
    n_same = 0
    n = 0
    for key in mask1.keys():
        n += mask1[key].numel()
        n_same += torch.eq(mask1[key], mask2[key]).sum().item()
    n_diff = n - n_same
    return n_diff

class PLATON(Algorithm):
    def __init__(self, 
                 max_train_steps, 
                 beta1=0.85, 
                 beta2=0.85,
                 initial_ratio=1.0, 
                 final_ratio=0.1,
                 initial_warmup_steps=1,
                 final_warmup_steps=1, 
                 deltaT=10,
                 non_mask_name=None,
                 mask_update_log_interval=None,):
        
        self.final_ratio_mask_after_initial_warmup = None
        self.last_final_ratio_mask = None
        self.mask_update_log_interval = mask_update_log_interval

        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

        self.max_train_steps = max_train_steps

        self.beta1 = beta1
        self.beta2 = beta2

        self.final_ratio = final_ratio
        self.initial_ratio = initial_ratio
        self.initial_warmup_steps = initial_warmup_steps
        self.final_warmup_steps = final_warmup_steps
        self.deltaT = deltaT

        self.non_mask_name_pattern = re.compile("|".join(non_mask_name), re.IGNORECASE) if non_mask_name is not None else None

        cubic_prune_start = initial_warmup_steps
        cubic_prune_end = max_train_steps - final_warmup_steps

        if not (cubic_prune_start < cubic_prune_end <= max_train_steps):
            print ("cubic_prune_start:", cubic_prune_start)
            print ("cubic_prune_end:", cubic_prune_end)
            print ("max_train_steps:", max_train_steps)
            raise ValueError("cubic_prune_start < cubic_prune_end <= max_train_steps must be satisfied, but got False")

        self.cubic_prune_start = cubic_prune_start
        self.cubic_prune_end = cubic_prune_end
        self.ratio_scheduler_steps = cubic_prune_end - cubic_prune_start
        
    @classmethod
    def from_args(self, max_train_steps, train_dataloader_len, args):
        initial_warmup_steps = _convert_timestr_to_int(args.initial_warmup_steps, max_train_steps, train_dataloader_len)
        final_warmup_steps = _convert_timestr_to_int(args.final_warmup_steps, max_train_steps, train_dataloader_len)
        deltaT = _convert_timestr_to_int(args.deltaT, max_train_steps, train_dataloader_len)
        mask_update_log_interval = _convert_timestr_to_int(args.mask_update_log_interval, max_train_steps, train_dataloader_len) if args.mask_update_log_interval is not None else None
        return self(
            max_train_steps=max_train_steps, 
            beta1=args.beta1, 
            beta2=args.beta2,
            initial_ratio=args.initial_ratio, 
            final_ratio=args.final_ratio,
            initial_warmup_steps=initial_warmup_steps, 
            final_warmup_steps=final_warmup_steps,
            deltaT=deltaT,
            non_mask_name=args.non_mask_name,
            mask_update_log_interval=mask_update_log_interval,
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


    def update_ipt_with_local_window(self, model, train_step_index):
        # Calculate the sensitivity and uncertainty 
        for n,p in model.named_parameters():
            if self.whether_mask_para(n):
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.ipt[n] = torch.zeros_like(p)
                    if self.beta2>0 and self.beta2!=1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                
                local_step = train_step_index % self.deltaT
                update_step = train_step_index // self.deltaT
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

    def calculate_mask_threshold(self, model, ratio):
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
        mask = self.create_mask(model, mask_threshold, is_dict)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    p.masked_fill_(mask[n], 0.0)
        return mask_threshold, mask

    def update_and_pruning(self, model, train_step_index):
        # Update importance score after optimizer stepping
        self.update_ipt_with_local_window(model, train_step_index)
        # Get the remaining ratio
        ratio, mask_ind = self.cubic_remaining_ratio_scheduler(train_step_index)
        if mask_ind and ratio < 1.0:
            # Mask weights during masking horizon
            mask_threshold, _ = self.mask_with_threshold(model, ratio)
        else:
            mask_threshold = None
        
        return ratio, mask_threshold
    
    def calculate_relative_sparsity(self, model):
        n_params = 0
        n_masked_params = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.whether_mask_para(n):
                    n_params += p.numel()
                    n_masked_params += p.eq(0.0).sum().item()
        return n_masked_params/n_params

    def print_pruning_modules(self, model):
        print ("list of model modules to be pruned:")
        for n, _ in model.named_parameters():
            if self.whether_mask_para(n):
                print (n)

    def match(self, event, state):
        return event in [Event.FIT_START, Event.BATCH_END, Event.FIT_END]

    def apply(self, event, state, logger):
        if event == Event.FIT_START:
            self.print_pruning_modules(state.model)
        elif event == Event.BATCH_END:
            if self.mask_update_log_interval is not None:
                if state.timestamp.batch.value == self.cubic_prune_start:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                    self.final_ratio_mask_after_initial_warmup = self.create_mask(state.model, mask_threshold, is_dict)
                if state.timestamp.batch.value > self.cubic_prune_start and state.timestamp.batch.value % self.mask_update_log_interval == 0:
                    mask_threshold, is_dict = self.calculate_mask_threshold(state.model, self.final_ratio)
                    updated_final_ratio_mask = self.create_mask(state.model, mask_threshold, is_dict)
                    n_reselection1 = _calculate_n_reselection(self.final_ratio_mask_after_initial_warmup, updated_final_ratio_mask)
                    if self.last_final_ratio_mask is not None:
                        n_reselection2 = _calculate_n_reselection(self.last_final_ratio_mask, updated_final_ratio_mask)
                        logger.log_metrics({"n_reselection_wrt_last_mask": int(n_reselection2)})
                    self.last_final_ratio_mask = updated_final_ratio_mask
                    logger.log_metrics({"n_reselection_wrt_initial_warmup": int(n_reselection1)})
            ratio, mask_threshold = self.update_and_pruning(state.model, state.timestamp.batch.value)
            logger.log_metrics({"remaining_ratio": float(ratio)})
            if mask_threshold is not None:
                logger.log_metrics({"mask_threshold": float(mask_threshold)})
        elif event == Event.FIT_END:
            relative_final_sparsity = self.calculate_relative_sparsity(state.model)
            logger.log_metrics({"relative_final_sparsity": float(relative_final_sparsity)})