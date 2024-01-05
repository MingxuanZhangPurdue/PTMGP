import torch

# calculate the magnitude mask threshold for a given sparsity
@torch.no_grad()
def calculate_magnitude_mask_threshold(model, sparsity, non_mask_name=None):
        
    # Calculate the final importance score
    is_dict = {}
    for n, p in model.named_parameters():
        if non_mask_name == None or not any(nd in n for nd in non_mask_name):
            is_dict[n] = p.abs().detach()

    # Calculate the mask threshold
    all_is = torch.cat([is_dict[n].view(-1) for n in is_dict])
    mask_threshold = torch.kthvalue(all_is, int(all_is.shape[0] * sparsity))[0].item()
    return mask_threshold

# create the mask for a given magnitude mask threshold
@torch.no_grad()
def create_magnitude_mask(model, non_mask_name=None, mask_threshold=0.0):

    mask = {}
    n_params = 0
    n_masked_params = 0
    for n, p in model.named_parameters():
        if non_mask_name == None or not any(nd in n for nd in non_mask_name):
            mask[n] = (p.data.abs() <= mask_threshold)
            n_params += p.numel()
            n_masked_params += mask[n].sum().item()

    return mask, n_masked_params/n_params

# zero out weights for a given mask
@torch.no_grad()
def prune_masked_weights(model, mask, non_mask_name=None):

    for n, p in model.named_parameters():
        if non_mask_name == None or not any(nd in n for nd in non_mask_name):
            p.data.masked_fill_(mask[n], 0.0)