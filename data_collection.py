from collections import deque, namedtuple
import random
import torch


#Will be how a single transition of environment is stored
Transition = namedtuple('Transition',
                        ['state',
                         'action', 
                         'next_state',  
                         'reward', 
                         'done'])


class MemoryBuffer:
    """A simple replay FIFO replay buffer"""

    def __init__(self, buffer_length : int, sample_size : int, random : random) -> None:
        self._memory = deque(maxlen=buffer_length)
        self._sample_size = sample_size
        self._random = random

    def sample(self) -> Transition:
        """Will randomly sample a batch of transitions from the replay buffer"""
        if len(self._memory) <= self._sample_size:
            sample = self._memory #sample will be a list of transitions
        else:
            sample = self._random.sample(self._memory, self._sample_size)

        state, action, next_state, reward, done = zip(*sample) #unpack list and create tuples of each data point in transition
        return Transition(state = torch.stack(state, dim = 0), #Each element of transition is the batch of values
                          action = torch.stack(action, dim = 0),
                          next_state = torch.stack(next_state, dim = 0),
                          reward = torch.stack(reward, dim = 0),
                          done = torch.stack(done, dim = 0))
    
    def add_data(self, trans : Transition) -> None:
        """Will add the data from the single transition into the buffer"""
        self._memory.append(trans)
