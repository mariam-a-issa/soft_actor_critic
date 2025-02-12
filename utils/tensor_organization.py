import torch

def group_to_boundaries_torch(groups : torch.Tensor):
    if len(groups) == 0:
        return torch.tensor([], dtype=torch.long)

    groups = groups.type(torch.long)
    
    # Find indices where the group changes
    change_indices = torch.where(groups[:-1] != groups[1:])[0] + 1

    # Always add the first index (0) and the last index (length of groups)
    boundaries = torch.cat([torch.tensor([0]), change_indices, torch.tensor([len(groups)])])

    return boundaries


