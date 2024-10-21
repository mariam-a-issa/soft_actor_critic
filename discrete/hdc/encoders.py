from math import pi

import torch
from torch import Tensor, device

from utils import MAX_ROWS #Maximum amount of rows of same vector length. (Basically can handle 50 devices). Same variable in hdc_implementation

class EXPEncoder:
    """Represents the exponential encoder from the hdpg_actor_critic code but changed to only use a tensors from pytorch"""
    def __init__(self, input_size : int, hyper_dim : int, dynamic : bool) -> None:
        """Will create an encoder that will work for vectors that have d dimensionality"""
        
        if dynamic:
            input_size = input_size * MAX_ROWS
        
        self._s_hdvec = torch.randn(input_size, hyper_dim, dtype=torch.float32) 
        self._bias = 2 * pi * torch.rand(hyper_dim, dtype=torch.float32)
        self._padding = dynamic #Will need to pad if dealing with dynamic state space
        self._input_size = input_size
        
    def __call__(self, state : Tensor | list[Tensor]) -> Tensor:
        """Will return the encoder hypervector. State needs the same dimensionality that was used to create the encoder"""
        
        if self._padding:
            state = _pad(state, self._input_size)
        
        if len(state.shape) == 1:
            return torch.exp(1j * (state @ self._s_hdvec + self._bias))
        
        #matmul with broadcast batch but need to unsqueeze so it does this instead of regular matmul
        return torch.exp(1j * ((state.unsqueeze(dim=1) @ self._s_hdvec).squeeze(dim=1) + self._bias)) #need to squeeze dim 1 to go from b_dim x 1 x hyper_v_dim -> b_dim x hyper_dim
        
    def to(self, dev : device) -> None:
        self._s_hdvec.to(dev)
        self._bias.to(dev)

class RBFEncoder:
    def __init__(self, input_size : int, hyper_dim : int, dynamic : bool):
        
        if dynamic:
            input_size = input_size * MAX_ROWS

        self._s_hdvec = torch.randn(input_size, hyper_dim, dtype=torch.float32) 
        self._bias = 2 * pi * torch.rand(hyper_dim, dtype=torch.float32)
        self._padding = dynamic
        self._input_size = input_size

    def __call__(self, state: Tensor | list[Tensor]) -> torch.Tensor:
        if self._padding:
            state = _pad(state, self._input_size)

        if len(state.shape) == 1:
            state = state @ self._s_hdvec + self._bias
            return torch.cos(state)

        #matmul with broadcast batch but need to unsqueeze so it does this instead of regular matmul
        state = (state.unsqueeze(dim=1) @ self._s_hdvec).squeeze(dim=1) + self._bias 
        return torch.cos(state)

    def to(self, dev : device) -> None:
        self._s_hdvec.to(dev)
        self._bias.to(dev)

def _pad(state : Tensor | list[Tensor], input_size : int) -> Tensor:
    """Will pad the state tensor correctly. Note that here we consider either a single unpadded vector or a batch list of unpadded vectors. 
    There will be no matrix input like traditional RL but output can be a matrix"""
    if isinstance(state, Tensor) and len(state.shape) == 1:
        padded_state = torch.zeros(input_size)
        padded_state[:len(state)] = state
        return padded_state
    elif isinstance(state, list):
        
        #Is slow but will flatten each tensor
        for i, s in enumerate(state):
            state[i] = s.view(-1)
            
        padded_state = torch.zeros(input_size) #Need to pad first one to desired length
        padded_state[:len(state[0])] = state[0]
        state[0] = padded_state
        return torch.nn.utils.rnn.pad_sequence(state, batch_first=True, padding_value=0)
    else:
        raise TypeError("Incorrect state representation for padding")
        
    