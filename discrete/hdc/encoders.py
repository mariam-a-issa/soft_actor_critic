from math import pi

import torch
from torch import Tensor, device


class EXPEncoder:
    """Represents the exponential encoder from the hdpg_actor_critic code but changed to only use a tensors from pytorch"""
    def __init__(self, seed_base : Tensor, seed_bias : Tensor) -> None:
        """Will create an encoder that will work for vectors that have d dimensionality"""
        
        self._s_hdvec = seed_base
        self._bias = seed_bias

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the encoder hypervector. State needs the same dimensionality that was used to create the encoder"""

        if len(state.shape) == 1:
            return torch.exp(1j * (state @ self._s_hdvec + self._bias))
        
        #Only batches of b_dim x v_dim
        assert len(state.shape) == 2
        
        #matmul with broadcast batch but need to unsqueeze so it does this instead of regular matmul
        return torch.exp(1j * ((state.unsqueeze(dim=1) @ self._s_hdvec).squeeze(dim=1) + self._bias)) #need to squeeze dim 1 to go from b_dim x 1 x hyper_v_dim -> b_dim x hyper_dim
    
    def to(self, dev : device) -> None:
        self._s_hdvec.to(dev)
        self._bias.to(dev)

class RBFEncoder:
    def __init__(self, seed_base : Tensor, seed_bias : Tensor):

        self._s_hdvec = seed_base
        self._bias = seed_bias

    def __call__(self, v: torch.Tensor) -> torch.Tensor:

        if len(v.shape) == 1:
            v = v @ self._s_hdvec + self._bias
            return torch.cos(v)
        
        #Only batches of b_dim x v_dim
        assert len(v.shape) == 2

        #matmul with broadcast batch but need to unsqueeze so it does this instead of regular matmul
        v = (v.unsqueeze(dim=1) @ self._s_hdvec).squeeze(dim=1) + self._bias 
        return torch.cos(v)
    
    def to(self, dev : device) -> None:
        self._s_hdvec.to(dev)
        self._bias.to(dev)