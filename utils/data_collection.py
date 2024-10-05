from collections import deque
from typing import NamedTuple
import random
import torch

class Transition(NamedTuple):
    "Will be how a single transition of environment is stored"
    state : torch.Tensor
    action : torch.Tensor
    next_state : torch.Tensor
    reward : float
    done : bool
    num_devices : int = None
    num_devices_n : int = None


class MemoryBuffer:
    """A simple replay buffer"""

    def __init__(self, buffer_length : int, sample_size : int, random : random, dynamic : bool) -> None:
        self._memory = deque(maxlen=buffer_length)
        self._sample_size = sample_size
        self._random = random
        self._dynamic = dynamic

    def sample(self) -> Transition:
        """Will randomly sample a batch of transitions from the replay buffer"""
        if len(self._memory) <= self._sample_size:
            sample = self._memory #sample will be a list of transitions
        else:
            sample = self._random.sample(self._memory, self._sample_size)

        state, action, next_state, reward, done, num_devices, num_devices_n = zip(*sample) #unpack list and create tuples of each data point in transition
        
        action = torch.stack(action, dim = 0)
        reward = torch.stack(reward, dim =0)
        done = torch.stack(done, dim = 0)
        num_devices = torch.stack(num_devices, dim = 0)
        num_devices_n = torch.stack(num_devices_n, dim = 0)
        
        if self._dynamic: #Need to do this since torch.utils.rnn.pad_sequence takes list of Tensors, stack can not have variable length of tensor
            state = list(state.flatten())
            next_state = list(next_state.flatten())
        else:
            state = torch.stack(state, dim=0)
            next_state = torch.stack(next_state, dim=0)
        
        return Transition(state=state, action=action, next_state=next_state, reward=reward, done=done, num_devices=num_devices, num_devices_n=num_devices_n)
    
    def add_data(self, trans : Transition) -> None:
        """Will add the data from the single transition into the buffer"""
        
        if len(trans.state.shape) == 2 and trans.num_devices is None:
            trans.num_devices = trans.state.shape[0]
            trans.num_device_n = trans.next_state.shape[0]
            trans.state = trans.state.flatten()
            trans.next_state = trans.next_state.flatten()
        
        self._memory.append(trans)
