import torch
from torch import Tensor

def reshape(logits : Tensor, batch_index : Tensor, action_dim : int, filler_val : float = -1e8) -> Tensor:
    """Will reshape the logits from the form bmxa to bxma where m is the variable about of devices. Since it is variable, the rows will be padded with zeros when necessary
    
    embed_states: a bmxe matrix where b is the batch size, m is the variable size of devices in each part group of the batch and e is the embeding dimension
    batch_index: bmx1, each element is the group that the element in the corresponding embed_state belongs to
        
    return: bxma matrix
    
    """
    a = action_dim
    b = batch_index.unique().numel()
    max_d = torch.bincount(batch_index).max().item()

    # Step 2: Compute positions within each group
    device_counts = torch.zeros(b, dtype=torch.long).scatter_add_(
        0, batch_index, torch.ones_like(batch_index)
    )
    device_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.long), device_counts[:-1]]), dim=0
    )

    device_indices = (torch.arange(len(batch_index)) - device_offsets[batch_index]).long()

    # Step 3: Prepare an output tensor with zeros
    output : Tensor = torch.zeros((b, max_d * a), dtype=logits.dtype) + filler_val
    # Step 4: Place actions into the reshaped matrix
    row_indices = batch_index
    col_indices = (device_indices[:, None] * a + torch.arange(a)).flatten()

    return output.index_put_((row_indices.repeat_interleave(a), col_indices), logits.flatten())

def generate_counting_tensor(ranges : Tensor) -> Tensor:
    """Will expand the ranges into indices counting from 0 to (a-b) where a and b are corresponding pairs in the array

    Args:
        ranges (Tensor): The ranges in array where each corresponding pair a, b represnts a range [a, b)

    Returns:
        Tensor: The expanded ranges
    """
    diffs = ranges[1:] - ranges[:-1]  # Compute step sizes
    # Generate counting sequences using broadcasting
    max_len = diffs.max().item()  # Find the longest sub-range
    range_tensor = torch.arange(max_len).expand(len(diffs), max_len)  # Expand a base range
    mask = range_tensor < diffs.unsqueeze(1)  # Mask out values beyond each range length
    return range_tensor[mask]

def permute_rows_by_shifts(matrix : Tensor, shifts : Tensor) -> Tensor:
    """Will permute each row of matrix according the the number of shifts found at the corresponding element in the shift array

    Args:
        matrix (Tensor): The N x D matrix whos rows will be shifted
        shifts (Tensor): The N dimension array whose elements correspond to the amount of shifts to take place in the matrix

    Returns:
        Tensor: The matrix whos rows are shifted
    """
    N, M = matrix.shape  # Get matrix dimensions
    indices = torch.arange(M).view(1, M).expand(N, M)  # Create base indices for rows
    shifted_indices = (indices - shifts.unsqueeze(1)) % M  # Apply shifts (negative for right shift)
    
    #Manipulation of memory with temp may not be needed?
    temp = matrix.gather(1, shifted_indices)  # Gather before writing
    matrix.copy_(temp)

    del temp

    return matrix  # Gather new indices

def permute_rows_by_shifts_matrix(matrix : Tensor, shifts: Tensor):
    """
    Permutes each sub-matrix of shape (B, D) within the 3D tensor `matrix` along the last dimension.

    Parameters:
    - matrix: Tensor of shape (N, B, D), where shifts are applied along D.
    - shifts: 1D tensor of shape (N,) containing shift values for each (B, D) block.

    Returns:
    - A new tensor with rows permuted accordingly.
    """
    N, B, D = matrix.shape  # Get matrix dimensions
    indices = torch.arange(D).view(1, D).expand(N, D)  # Create base indices for last dimension
    shifted_indices : Tensor = (indices - shifts.unsqueeze(1)) % D  # Apply shifts (negative for right shift)

    # Expand indices to match (N, B, D) for proper broadcasting
    shifted_indices = shifted_indices.unsqueeze(1).expand(-1, B, -1)

    # Generate batch indices to maintain correct selection
    batch_indices = torch.arange(N).view(N, 1, 1).expand(-1, B, -1)
    
    return matrix[batch_indices, torch.arange(B).view(1, B, 1).expand(N, -1, -1), shifted_indices]

def generate_batch_index(state_index : Tensor):
    """Exapnds the ranges so that each subsequent range contains a subsuquent number repeated by the length of the range

    Args:
        state_index (Tensor): The ranges in array where each corresponding pair a, b represnts a range [a, b)

    Returns:
        Tensor: The ranges expanded according to the correct format
    """
    diffs = state_index[1:] - state_index[:-1]  # Compute segment sizes

    # Create repeated indices in a fully vectorized manner
    batch_index = torch.arange(len(diffs)).repeat_interleave(diffs)

    return batch_index