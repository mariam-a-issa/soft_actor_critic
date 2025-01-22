import torch
import math

def positional_encoding(pos, dim):
    w = torch.exp(torch.arange(0, dim, 2, device=pos.device) * (-math.log(10000.0) / dim))
    pos_w = pos.unsqueeze(1) * w

    pe = torch.zeros(len(pos), dim, device=pos.device)
    pe[:, 0::2] = torch.sin(pos_w)
    pe[:, 1::2] = torch.cos(pos_w)

    return pe

