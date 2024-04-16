from math import pi

import torch
from torch import Tensor
from torch import device


class EXPEncoder:
    """Represents the exponential encoder from the hdpg_actor_critic code but changed to only use a tensors from pytorch"""
    def __init__(self, d : int, dev : device):
        """Will create an encoder that will work for vectors that have d dimensionality"""
        
        self._s_hdvec = torch.randn(d, dtype=torch.float32, device=dev).unsqueeze(0)
        for _ in range(d - 1):
            self._s_hdvec = torch.cat((self._s_hdvec, torch.rand(d, dtype=torch.float32, device=dev).unsqueeze(0)), dim = 0)
        
        self._bias = torch.randn(d, dtype=torch.float32, device=dev) * 2 * pi
        self.d = d #Will be used by functions to know how to create their models

    def __call__(self, v : Tensor) -> Tensor:
        """Will return the encoder hypervector. State needs the same dimensionality that was used to create the encoder"""

        #Same as line 53

        if len(v.shape) == 1:
            return torch.exp(1j * (torch.matmul(v, self._s_hdvec) + self._bias))
        
        #Only batches of b_dim x v_dim
        assert len(v.shape) == 2

        batch_dim = v.shape[0]

        new_v = v.unsqueeze(1) # b_dim x 1 x v_dim
        
        batch_vector = self._s_hdvec.repeat(batch_dim, 1, 1) #b_dim x v_dim x hyper_v_dim
        batch_bias = self._bias.repeat(batch_dim, 1) #b_dim x hyper_v_dim
        
        #bmm is batch matrix multiplication
        return torch.exp(1j * (torch.bmm(new_v, batch_vector).squeeze(1) + batch_bias)) #need to squeeze dim 1 to go from b_dim x 1 x hyper_v_dim -> b_dim x hyper_dim
    

class RBFEncoder:
    def __init__(self, in_size: int, out_size: int, dev : device):
        self._in_size = in_size
        self._out_size = out_size

        self._s_hdvec = torch.randn(in_size, out_size, dtype=torch.float32, device=dev) / in_size #Why normalize with in_size
        self._bias = 2 * pi * torch.randn(out_size, dtype=torch.float32, device=dev)
  

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        
        #TODO potentially look at torch.no_grad in here or when called 

        if len(v.shape) == 1:
            v = v @ self._s_hdvec + self._bias
            v = torch.cos(v)
            return v
        
        #Only batches with 
        assert len(v.shape) == 2

        batch_dim = v.shape[0]
        
        new_v = v.unsqueeze(1) # b_dim x 1 x v_dim
        
        batch_vector = self._s_hdvec.repeat(batch_dim, 1, 1) #b_dim x v_dim x hyper_v_dim
        batch_bias = self._bias.repeat(batch_dim, 1) #b_dim x hyper_v_dim
        
        return torch.cos(torch.bmm(new_v, batch_vector).squeeze(1) + batch_bias)