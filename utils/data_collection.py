from collections import deque
from typing import NamedTuple
import random

import torch
from torch_geometric.data import Batch
from torch import tensor, Tensor
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

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
    next_state_index : Tensor = None,
    priority_index : Tensor = None


class MemoryBuffer:
    """A simple replay buffer"""

    def __init__(self, buffer_length : int, sample_size : int) -> None:
        self._memory = deque(maxlen=buffer_length)
        self._sample_size = sample_size

    def sample(self) -> Transition:
        """Will randomly sample a batch of transitions from the replay buffer"""
        if len(self._memory) <= self._sample_size:
            sample = self._memory #sample will be a list of transitions
        else:
            sample = random.sample(self._memory, self._sample_size)

        state, action, next_state, reward, done, _ , _, _, _, _ = zip(*sample) #unpack list and create tuples of each data point in transition
        
        action = torch.stack(action, dim = 0)
        reward = torch.stack(reward, dim =0)
        done = torch.stack(done, dim = 0)
        
        state = torch.stack(state, dim=0)
        next_state = torch.stack(next_state, dim=0)
        
        return Transition(state=state, action=action, next_state=next_state, reward=reward, done=done)
    
    #We should store as a matrix instead of a flattened vector so that certain models/encoders can access each individual host however they choose
    #May want to fix in future so that we do not have num devices as this data is redundant if we are storing as a matrix
    def add_data(self, trans : Transition) -> None:
        """Will add the data from the single transition into the buffer"""
        self._memory.append(trans)

class PrioritizedMemoryBuffer():

    def __init__(self, buffer_length : int, sample_size : int, alpha : float, beta : float, device : torch.device) -> None:
        self._memory = TensorDictPrioritizedReplayBuffer(
            alpha=alpha,         
            beta=beta,           
            eps=1e-6,           
            priority_key="td_error",             
            storage=LazyTensorStorage(max_size=buffer_length, device=device), 
            batch_size=sample_size,
        )

        self._sample_size = sample_size
        self._max_priority = torch.tensor(1.0, device=device)

    def sample(self) -> Transition:
        batch = self._memory.sample()

        return Transition(
            state=batch['state'],
            action=batch['action'],
            next_state=batch['next_state'],
            reward=batch['reward'],
            done=batch['done'],
            priority_index=batch['index']
        )

    def add_data(self, trans : Transition) -> None:
        #The transition contains tensors
        td = TensorDict(
            {'state' : trans.state,
             'action' : trans.action,
             'next_state' : trans.next_state,
             'reward' : trans.reward,
             'done' : trans.done,
             'td_error' : self._max_priority
            }
        )

        self._memory.add(td)

    def update_priority(self, trans : Transition, error : Tensor) -> None:

        self._max_priority = torch.max(self._max_priority, error.max())

        td = TensorDict(
            {'state' : trans.state,
             'action' : trans.action,
             'next_state' : trans.next_state,
             'reward' : trans.reward,
             'done' : trans.done,
             'td_error' : error,
             'index' : trans.priority_index
            },
            batch_size=trans.state.shape[0]
        )


        self._memory.update_tensordict_priority(td)



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

        state, action, next_state, reward, done, _, _, _, _, _ = zip(*sample)
        
        state_index = torch.tensor([0]+[state[i - 1].shape[0] for i in range(1, len(state))]+ [state[-1].shape[0]], device=state[0].device)
        next_state_index = torch.tensor([0]+[next_state[i - 1].shape[0] for i in range(1, len(next_state))]+ [next_state[-1].shape[0]], device=state[0].device)
        
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
    
    def __init__(self, buffer_length : int, sample_size : int) -> None:
        self._memory = deque(maxlen=buffer_length)
        self._sample_size = sample_size
        
    def sample(self) -> Transition:
        if len(self._memory) <= self._sample_size:
            sample = self._memory
        else:
            sample = random.sample(self._memory, self._sample_size)
            
        state, action, next_state, reward, done, _, _, _, _, _ = zip(*sample) #In this case state and next_state are tuples of Data
        cur_batch = Batch.from_data_list(state)
        next_batch = Batch.from_data_list(next_state)
        
        state_index = group_to_boundaries_torch(cur_batch.batch)
        next_state_index = group_to_boundaries_torch(next_batch.batch)
        
        action = torch.stack(action, dim = 0)
        reward = torch.stack(reward, dim =0)
        done = torch.stack(done, dim = 0)
        
        return Transition(state=cur_batch, next_state=next_batch, action=action, reward=reward, done=done, num_devices=None, num_devices_n=None, state_index=state_index, next_state_index=next_state_index)
    
    def add_data(self, trans : Transition) -> None:
        self._memory.append(trans)