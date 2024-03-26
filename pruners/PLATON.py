import torch
import re
from composer.core import Algorithm, Event
from pruners.utils_composer import _convert_timestr_to_int

class PLATON(Algorithm):
    def __init__(
        self,
        max_train_steps,
        beta1=0.85,
        beta2=0.85,
        initial_sparsity=0.0,
        final_sparsity=0.0,
        initial_warmup_steps=0,
        final_warmup_steps=0,
        pruning_interval=10,
        pruning_params=None
    ):
        
        assert initial_sparsity < 1.0 and initial_sparsity >= 0.0, "initial_sparsity must be in the range [0, 1)"
        assert final_sparsity < 1.0 and final_sparsity >= 0.0, "final_sparsity must be in the range [0, 1)"
        assert initial_sparsity <= final_sparsity, "initial_sparsity must be less than or equal to final_sparsity"  
        
        self.n_total_param_for_pruning = 0
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

        self.max_train_steps = max_train_steps

        self.beta1 = beta1
        self.beta2 = beta2

        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity

        pruning_start = initial_warmup_steps
        pruning_end = max_train_steps - final_warmup_steps - 1

        assert 0 <= pruning_start < pruning_end, (
            f"pruning_start: {pruning_start}, "
            f"pruning_end: {pruning_end}, "
            "Condition 0 <= pruning_start < pruning_end must be satisfied, but got False"
        )

        self.pruning_start = pruning_start
        self.pruning_end = pruning_end
        self.pruning_interval = pruning_interval
        self.pruning_params = re.compile("|".join(pruning_params), re.IGNORECASE) if pruning_params is not None else None
        
    @classmethod
    def from_args(
        self, 
        train_size, 
        max_train_steps, 
        train_dataloader_len, 
        args
    ):
        initial_warmup_steps = _convert_timestr_to_int(args.initial_warmup_steps, max_train_steps, train_dataloader_len)
        final_warmup_steps = _convert_timestr_to_int(args.final_warmup_steps, max_train_steps, train_dataloader_len)
        pruning_interval = _convert_timestr_to_int(args.pruning_interval, max_train_steps, train_dataloader_len)
        return self(
            max_train_steps=max_train_steps,
            beta1=args.beta1,
            beta2=args.beta2,
            initial_sparsity=args.initial_sparsity,
            final_sparsity=args.final_sparsity,
            initial_warmup_steps=initial_warmup_steps,
            final_warmup_steps=final_warmup_steps,
            pruning_interval=pruning_interval,
            pruning_params=args.pruning_params
        )

    def whether_prune_param(self, n):
        if self.pruning_params == None:
            return True
        else:
            return bool(re.search(self.pruning_params, n))

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
            print ("Pruning has ended, generate the final mask")
        else:
            raise ValueError(f"Invalid train_step_index value: {train_step_index}")
        return sparsity, pruning_ind


    def update_ipt_with_local_window(self, model, train_step_index):
        for n,p in model.named_parameters():
            if self.whether_prune_param(n):
                if n not in self.exp_avg_ipt:
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.ipt[n] = torch.zeros_like(p)
                    if self.beta2>0 and self.beta2!=1:
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                
                local_step = train_step_index % self.pruning_interval
                update_step = train_step_index // self.pruning_interval
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

    def calculate_mask_threshold(self, model, sparsity):
        is_dict = {}
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                if self.beta2 > 0 and self.beta2<1:
                    is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc[n] 
                elif self.beta2 == 1.:
                    is_dict[n] = self.exp_avg_ipt[n]
                elif self.beta2 == 2.:
                    is_dict[n] = self.exp_avg_ipt[n] * self.exp_avg_unc.sqrt()
                else:
                    is_dict[n] = self.exp_avg_ipt[n] * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
        all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
        mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0]*(sparsity)))[0].item()
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

    def update_and_pruning(self, model, train_step_index):
        self.update_ipt_with_local_window(model, train_step_index)
        sparsity, pruning_ind = self.sparsity_scheduler(train_step_index)
        if pruning_ind and sparsity > 0.0:
            mask_threshold, _ = self.prune_with_threshold(model, sparsity)
        else:
            mask_threshold = None
        return sparsity, mask_threshold

    def print_pruning_modules(self, model):
        print ("List of model modules to be pruned:")
        for n, p in model.named_parameters():
            if self.whether_prune_param(n):
                print (n)
                self.n_total_param_for_pruning += p.numel()
        print ("Total number of candidate parameters for pruning:", self.n_total_param_for_pruning)

    def match(self, event, state):
        return event in [Event.FIT_START, Event.BATCH_END]

    def apply(self, event, state, logger):
        if event == Event.FIT_START:
            self.print_pruning_modules(state.model)
        elif event == Event.BATCH_END:
            sparsity, mask_threshold = self.update_and_pruning(state.model, state.timestamp.batch.value-1)
            if (logger is not None):
                logger.log_metrics({"sparsity": float(sparsity)})
                if mask_threshold is not None:
                    logger.log_metrics({"mask_threshold": float(mask_threshold)})