import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from collections import namedtuple

V_HIDDEN_LAYER = 12
Q_HIDDEN_SIZE = 12
ACTOR_HIDDEN_SIZE = 12
STEP_V = .01
STEP_Q = .01
STEP_ACTOR = .01
TAU = .1
V_GAMMA = .1

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'next_action', 'reward']) #If bad performance just switch to a tensor


#TODO How to make it a batch for the updates?


class BaseNN(nn.Module):
    """Base class for constructing NNs"""

    def __init__(self, input_size : int,  outpute_size : int, hidden_size : int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = outpute_size
        self.hidden_size = hidden_size

        self._layers = nn.Sequential(
                                nn.Linear(self.input_size, self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x : Tensor) -> Tensor:
        return self._layers(x)

class TargetValueModel(BaseNN):
    """Target Value Function that contains method to update parameters"""

    def __init__(self, *args, v_func : BaseNN) -> None:
        super().__init__(*args)
        self._v_func = v_func

    def update_parameters(self, trans : Transition) -> None:
        with torch.no_grad():
            v_params = self._v_func.parameters()
            for param in self.parameters():
                new_value = TAU * next(v_params) + (1 - TAU)(param)
                param.copy_(new_value)


class QModel(BaseNN):
    """Will be the model representing a q function"""

    def __init__(self, *args, target_v_func : TargetValueModel, optimizer : optim) -> None:
        super().__init__(*args)
        self._v_func = target_v_func
        self._optim : optim.Adam = optimizer(self.parameters())
    
    def update_parameters(self, trans : Transition) -> None:
        self._optim.zero_grad()
        grad_val = self(trans.state)[trans.action](self(trans.state)[trans.action] - trans.reward - V_GAMMA * self._v_func(trans.next_state))
        grad_val.backward()
        self._optim.step()


class QFunction:
    """Will contain two q models that will be used as a single q function."""
    def __init__(self, q1 : QModel,  q2 :  QModel) -> None:
        self._q1 = q1
        self._q2 = q2

    def forward(self, x : Tensor) -> Tensor:
        #TODO implement so that its minimum for a speciifc action
        torch.min(self._q1(x), self._q2(x))

    def update_parameters(self, trans : Transition):
        for q in self._list_q_funcs():
            q.update_parameters(trans)

    def _list_q_funcs(self) -> list[QModel]:
        return [self._q1, self._q2]

class Actor(BaseNN):
    pass


def train(gm, len_state : int , len_output : int) -> None:
    """"
    init all nns and put into list ordered according to order of updates in paper

    for each episode:
        get transition sequence
        add to memory

    if gradient step time
        for nn in nn_list:
            nn.update_parameters(batch of transitions)

    """
    
