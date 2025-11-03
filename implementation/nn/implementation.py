from copy import deepcopy

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F
from torch import nn

from .architecture import BaseNN
from utils import LearningLogger

#Parameter update implementation from https://arxiv.org/abs/1910.07207


class QFunction(nn.Module):

    def __init__(self, 
                 input_size : int, 
                 output_size : int, 
                 hidden_size : int) -> None:
        
        """Will create a q function that will use two q models"""
        super().__init__() 
        self._q1 = BaseNN(input_size, output_size, [hidden_size, hidden_size], id=1)
        self._q2 = BaseNN(input_size, output_size, [hidden_size, hidden_size], id=2)
        
        self._action_s = output_size
        self._state_s = input_size

    def q_values(self, state : Tensor) -> Tensor:
        return self._q1(state), self._q2(state)

    def forward(self, state : Tensor) -> Tensor:
        """Will give a Tensor where each index represents the q value for the corresponding action"""

        q1 = self._q1(state)
        q2 = self._q2(state)

        description = 'Q1'
        LearningLogger().log_scalars({f'{description} Mean' : q1.mean(), 
                            f'{description} Max' : float(q1.max()),
                            f'{description} Min' : float(q1.min()),
                            f'{description} Std' : float(q1.std())}, steps=LearningLogger().cur_step())
        
        description = 'Q2'
        LearningLogger().log_scalars({f'{description} Mean' : q2.mean(), 
                            f'{description} Max' : float(q2.max()),
                            f'{description} Min' : float(q2.min()),
                            f'{description} Std' : float(q2.std())}, steps=LearningLogger().cur_step())

        return torch.min(self._q1(state), self._q2(state))
        
class QFunctionTarget:

    def __init__(self, actual : QFunction, tau : float) -> None:
        self._actual = actual

        if actual is not None:
            self._target = deepcopy(actual)

        self._tau = tau

    def set_actual(self, actual : QFunction) -> None:
        """Will set the actual if it was not set in init"""
        self._actual = actual
        self._target = deepcopy(actual)

    def to(self, device) -> None:
        self._actual.to(device)
        self._target.to(device)

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the q values from the target network"""
        return self._target(state)
    
    def update(self) -> None:
        """Will do polyak averaging to update the target"""
        for param, target_param in zip(self._actual._q1.parameters(), self._target._q1.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._actual._q2.parameters(), self._target._q2.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)


class Actor(BaseNN):

    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 hidden_size : int) -> None:
            
        super().__init__(input_size, output_size, [hidden_size, hidden_size])
        self._action_s = output_size
        self._action_act_s = output_size
        
        self._state_s = input_size
        self._state_act_s = input_size

    def logits(self, state : Tensor) -> Tensor:
        return super().forward(state)

    def forward(self, state : Tensor) -> tuple[Tensor]:
        """Will give the action, log_prob, and action_probs of action"""

        #Implementation very similar to cleanrl
        logits : Tensor = self.logits(state)
            
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_probs = dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, log_prob, action_probs
    
    # We are not trying to do padding here
    # def _mask_func(self, batch_size : int, logits : Tensor, mask_num : float, num_devices : Tensor) -> Tensor:
    #     # Create an index tensor for each row, broadcast to match the size of matrix         # [1, 2, 3, ... i]
    #     row_indices = torch.arange(logits.size(-1)).unsqueeze(0).expand(batch_size, -1)      # [1, 2, 3  ... i]
    #     # Use broadcasting to create a boolean mask                                          # ^  
    #     num_devices *= self._action_s                                                        # |
    #     mask = row_indices < num_devices.unsqueeze(1)                                        # |_ Then create a mask of same dimensions as this matrix where True at indicies are less than action size per device times device 
    #     return logits.masked_fill(~mask, float(mask_num))
    
    def sample(self, state : Tensor) -> Tensor:
        return self(state)[0]

    def evaluate(self, state : Tensor) -> Tensor:
        """Will return the best action for evaulation"""
        
        return torch.argmax(self.logits(state))
