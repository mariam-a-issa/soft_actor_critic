import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.distributions import MultivariateNormal
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
    

class ValueFunction(BaseNN):
    """Will be target function that updates parameters according to equation 6"""

    def __init__(self, *args, q_function : 'QFunction', actor : 'Actor') -> None:
        super().__init__(*args)
        self._optim = optim.Adam(self.parameters())
        self._actor = actor
        self._q_function = q_function

    def update_parameters(self, trans : Transition) -> None:
        val = self(trans.state) * (float(self(trans.state)) - self._q_function(trans.state, trans.action) + self._actor(trans.state)[1]) #actor returns action, then log prob
        self._optim.zero_grad()
        val.backward()
        self._optim.step()


class TargetValueModel(BaseNN):
    """Target target value function that contains method to update parameters not using gradient according to psuedocode""" #Is the psuedocode representation the exponentially moving average?

    def __init__(self, *args, v_func : ValueFunction) -> None:
        super().__init__(*args)
        self._v_func = v_func

    def update_parameters(self, trans : Transition) -> None:
        with torch.no_grad():
            v_params = self._v_func.parameters()
            for param in self.parameters():
                new_value = TAU * next(v_params) + (1 - TAU)(param)
                param.copy_(new_value)


class QModel(BaseNN):
    """Will be the model representing a q function.
    Its input should be the state and the action concatenated.
    Updated according to the equation 9"""

    def __init__(self, *args, target_v_func : TargetValueModel) -> None:
        super().__init__(*args)
        self._v_func = target_v_func
        self._optim = optim.Adam(self.parameters())
    
    def update_parameters(self, trans : Transition) -> None:
        input_tensor = torch.cat((trans.state, trans.action), dim = 0)
        val = self(input_tensor) * float(self(input_tensor) - trans.reward - V_GAMMA * self._v_func(trans.next_state))
        
        self._optim.zero_grad()
        val.backward()
        self._optim.step()


class QFunction:
    """Will contain two q models that will be used as a single q function."""

    def __init__(self, q1 : QModel,  q2 :  QModel) -> None:
        self._q1 = q1
        self._q2 = q2

    def forward(self, x : Tensor) -> Tensor:
        """x should be the state and the action concatenated on top of each other"""
        torch.min(self._q1(x), self._q2(x))

    def update_parameters(self, trans : Transition) -> None:
        for q in self._list_q_funcs():
            q.update_parameters(trans)

    def _list_q_funcs(self) -> list[QModel]:
        return [self._q1, self._q2]


class Actor(nn.Module):
    """Actor updated according to equation 13"""

    def __init__(self, input_size, output_size, q_func : QFunction) -> None:
        super().__init__()
        self._hidden1 = nn.Linear(input_size, 2)
        self._hidden2 = nn.Linear(2, 3)
        self._mean = nn.Linear(3, output_size)
        self._std = nn.Linear(3, output_size)
    
    def forward(self, x : Tensor) -> tuple[Tensor, Tensor]:
        
        mean, std = self._forward_gaussian(x)
        covar = torch.pow(torch.diag(std), 2) #Would defining the network to give std then covar work?

        #I assume this is a spherical guassian as covariance is not defined on values not on diagonal
        dist = MultivariateNormal(mean, covar)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def update_parameters(self, trans : Transition) -> None:
        pass
        """
        For gradients that are w.r.t the parameters:
        Do output.backward() get grad of each parameter and add to a tensor

        For gradients that are w.r.t action:
        Do action.requires_grad_() then find the grad of the action after ouput.backward()

        How will we multiple a vector of with dimension size of actor and of size parameters?
        """
    
    def _forward_gaussian(self, x : Tensor) -> tuple[Tensor, Tensor]:

        x = nn.functional.relu(self._hidden1(x))
        x = nn.functional.relu(self._hidden2(x))

        mean = self._mean(x)
        std = self._std(x)

        return mean, std
    
    #TODO make update parameters and sample from spherical guassian

    def _nosy_vector(self) -> Tensor:
        """Will create a vector epsilon sampled from a spherical guassian"""
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
    
