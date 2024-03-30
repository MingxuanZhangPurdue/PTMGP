import re
import torch
from composer.core import Algorithm, Event


def generate_mask(model, sparsity, pruned_modules=None):
    pattern =  re.compile("|".join(pruned_modules), re.IGNORECASE) if pruned_modules is not None else None
    is_dict = {}
    print ("Generating mask for pruned modules, with sparsity:", sparsity)
    for n, p in model.named_parameters():
        if pattern is None or bool(re.search(pattern, n)):
            is_dict[n] = p.detach().abs()
    all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
    mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * sparsity))[0].item()
    mask = {}
    for n, p in model.named_parameters():
        if pattern is None or bool(re.search(pattern, n)):
            mask[n] = (is_dict[n] < mask_threshold)
            # Make sure that the pruned weights are zeroed out
            p.detach().masked_fill_(mask[n], 0.0)
    print ("Mask threshold:", mask_threshold)
    return mask

class PatternLock(Algorithm):
    def __init__(
            self,
            mask,
        ):
        self.mask = mask
    
    def verify_pruned_model(self, model, mask):
        for n, p in model.named_parameters():
            if n in mask:
                assert (p.detach().masked_select(mask[n]).abs().sum().item() == 0.0)

    def zero_pruned_grad(self, model, mask):
        for n, p in model.named_parameters():
            if n in mask:
                p.grad.detach().masked_fill_(mask[n], 0.0)

    def print_pruned_modules(self, mask):
        n_total_param_pruned = 0
        print ("List of model modules pruned:")
        for n in mask.keys():
            print (n)
            n_total_param_pruned += mask[n].sum().item()
        print ("Total number of parameters pruned:", n_total_param_pruned)

    def match(self, event, state):
        return event in [Event.FIT_START, Event.AFTER_TRAIN_BATCH]

    def apply(self, event, state, logger):
        if event == Event.FIT_START:
            self.print_pruned_modules(self.mask)
            self.verify_pruned_model(state.model, self.mask)
        elif event == Event.AFTER_TRAIN_BATCH:
            self.zero_pruned_grad(state.model, self.mask)