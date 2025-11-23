from math import pi

import torch
from torch import Tensor, device


class EXPEncoder:
    """Represents the exponential encoder from the hdpg_actor_critic code but changed to only use a tensors from pytorch"""
    def __init__(self, input_size : int, hyper_dim : int) -> None:
        """Will create an encoder that will work for vectors that have d dimensionality"""
        
        self._s_hdvec = torch.randn(input_size, hyper_dim, dtype=torch.float32) 
        self._bias = 2 * pi * torch.rand(hyper_dim, dtype=torch.float32)
        self._input_size = input_size
        
    def __call__(self, state : Tensor | list[Tensor]) -> Tensor:
        """Will return the encoder hypervector. State needs the same dimensionality that was used to create the encoder"""
        
        #matmul with broadcast batch but need to unsqueeze so it does this instead of regular matmul
        return torch.exp(1j * ((state.unsqueeze(dim=1) @ self._s_hdvec).squeeze(dim=1) + self._bias)) #need to squeeze dim 1 to go from b_dim x 1 x hyper_v_dim -> b_dim x hyper_dim
        
    def to(self, dev : device) -> None:
        self._s_hdvec = self._s_hdvec.to(dev)
        self._bias = self._bias.to(dev)

class RBFEncoder:
    def __init__(self, input_size : int, hyper_dim : int):

        self._s_hdvec = torch.randn(input_size, hyper_dim, dtype=torch.float32) 
        self._bias = 2 * pi * torch.rand(hyper_dim, dtype=torch.float32)
        self._input_size = input_size

    def __call__(self, state: Tensor | list[Tensor]) -> torch.Tensor:

        if len(state.shape) == 1:
            state = state @ self._s_hdvec + self._bias
            return torch.cos(state)

        #matmul with broadcast batch but need to unsqueeze so it does this instead of regular matmul
        state = (state.unsqueeze(dim=1) @ self._s_hdvec).squeeze(dim=1) + self._bias 
        return torch.cos(state)

    def to(self, dev : device) -> None:
        self._s_hdvec = self._s_hdvec.to(dev)
        self._bias = self._bias.to(dev)
        
    