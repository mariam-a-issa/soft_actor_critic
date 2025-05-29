from collections import deque
from typing import NamedTuple
import random

import torch
from torch_geometric.data import Batch
from torch import tensor, Tensor

from .tensor_organization import group_to_boundaries_torch

class Transition(NamedTuple):
    "Will be how a single transition of environment is stored"
    state : torch.Tensor
    action : torch.Tensor
    next_state : torch.Tensor
    reward : float
    done : bool
    num_devices : int = None
    num_devices_n : int = None,
    state_index : Tensor = None,
    next_state_index : Tensor = None


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
            state = list(state)
            next_state = list(next_state)
        else:
            state = torch.stack(state, dim=0)
            next_state = torch.stack(next_state, dim=0)
        
        return Transition(state=state, action=action, next_state=next_state, reward=reward, done=done, num_devices=num_devices, num_devices_n=num_devices_n)
    
    #We should store as a matrix instead of a flattened vector so that certain models/encoders can access each individual host however they choose
    #May want to fix in future so that we do not have num devices as this data is redundant if we are storing as a matrix
    def add_data(self, trans : Transition) -> None:
        """Will add the data from the single transition into the buffer"""
        
        if len(trans.state.shape) == 2 and trans.num_devices is None:
            
            trans = Transition(state=trans.state,
                               action = trans.action,
                               next_state=trans.next_state,
                               reward=trans.reward,
                               done = trans.done,
                               num_devices=tensor(trans.state.shape[0]),
                               num_devices_n=tensor(trans.next_state.shape[0])
                               )
        
        self._memory.append(trans)


class DynamicMemoryBuffer():
    """Replay buffer where due to the dynamic size of the state, part of the state is collpased into the batch dim"""
    
    def __init__(self, buffer_size : int, sample_size : int) -> None:
        self._memory = deque(maxlen=buffer_size)
        self._sample_size = sample_size
    
    def sample(self) -> Transition:
        if len(self._memory) <= self._sample_size:
            sample = self._memory #sample will be a list of transitions
        else:
            sample = random.sample(self._memory, self._sample_size)

        state, action, next_state, reward, done, _, _, _, _ = zip(*sample)
        
        state_index = torch.tensor([0]+[state[i - 1].shape[0] for i in range(1, len(state))]+ [state[-1].shape[0]])
        next_state_index = torch.tensor([0]+[next_state[i - 1].shape[0] for i in range(1, len(next_state))]+ [next_state[-1].shape[0]])
        
        state_index = torch.cumsum(state_index, dim=0)
        next_state_index = torch.cumsum(next_state_index, dim = 0)
        
        state = torch.cat(state, dim=0)
        next_state = torch.cat(next_state, dim=0)
        
        action = torch.stack(action, dim = 0)
        reward = torch.stack(reward, dim =0)
        done = torch.stack(done, dim = 0)
        
        return Transition(state=state, action=action, next_state=next_state, reward=reward, done=done, num_devices=None, num_devices_n=None, state_index=state_index, next_state_index=next_state_index)    
    
    def add_data(self, trans : Transition) -> None:
        self._memory.append(trans)
            
        
class GraphMemoryBuffer():
    """A type of memory buffer that will retain graph represententatoins"""
    
    def __init__(self, buffer_length : int, sample_size : int, mask_subnet_state_index : bool = False) -> None:
        self._memory = deque(maxlen=buffer_length)
        self._sample_size = sample_size
        self._mask_subnet_state_index = mask_subnet_state_index
        
    def sample(self) -> Transition:
        if len(self._memory) <= self._sample_size:
            sample = self._memory
        else:
            sample = random.sample(self._memory, self._sample_size)
            
        state, action, next_state, reward, done, _, _, _, _ = zip(*sample) #In this case state and next_state are tuples of Data
        cur_batch = Batch.from_data_list(state)
        next_batch = Batch.from_data_list(next_state)
        
        if self._mask_subnet_state_index:
            cur_is_not_subnet = cur_batch.x[:, 0] != 1
            next_is_not_subnet = next_batch.x[:, 0] != 1
            
            cur_batch_batch = cur_batch.batch[cur_is_not_subnet]
            next_batch_batch = next_batch.batch[next_is_not_subnet]
        else:
            cur_batch_batch = cur_batch.batch
            next_batch_batch = next_batch.batch
        
        state_index = group_to_boundaries_torch(cur_batch_batch)
        next_state_index = group_to_boundaries_torch(next_batch_batch)
        
        action = torch.stack(action, dim = 0)
        reward = torch.stack(reward, dim =0)
        done = torch.stack(done, dim = 0)
        
        return Transition(state=cur_batch, next_state=next_batch, action=action, reward=reward, done=done, num_devices=None, num_devices_n=None, state_index=state_index, next_state_index=next_state_index)
    
    def add_data(self, trans : Transition) -> None:
        self._memory.append(trans)