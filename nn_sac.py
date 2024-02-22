import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.distributions import MultivariateNormal
from collections import namedtuple
from copy import deepcopy

STEP_V = .01
STEP_Q = .01
STEP_ACTOR = .01
TAU = .1
V_GAMMA = .1

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'next_action', 'reward']) #If bad performance just switch to a tensor


#TODO How to make it a batch for the updates?


class BaseNN(nn.Module):
    """Base class for constructing NNs"""

    def __init__(self, input_size : int,  outpute_size : int, hidden_size : int = 2) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = outpute_size
        self._hidden_size = hidden_size

        self.layers = nn.Sequential(
                                nn.Linear(self.input_size, self._hidden_size),
                                nn.ReLU(),
                                nn.Linear(self._hidden_size, self._hidden_size),
                                nn.ReLU(),
                                nn.Linear(self._hidden_size, self.output_size))

    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)
    

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


class Actor(BaseNN):
    """Actor updated according to equation 13"""

    def __init__(self, input_size : int, output_size : int) -> None:
        super().__init__(input_size, output_size * 2) #Needs to be twice as we need to get mean and covar vectors
        self.real_output_size = output_size

        #May need to pick the random numbers differently
        mean = torch.randn(self.real_output_size) 
        varience = float(torch.rand(1)) 

        covar = varience * torch.eye(self.real_output_size)
        self._s_dist = MultivariateNormal(mean, covar)
    
    def forward(self, x : Tensor) -> tuple[Tensor, Tensor]:
        
        mean, covar = self._forward_gaussian(x)

        dist = MultivariateNormal(mean, covar)

        action = dist.sample()
        action.requires_grad = True

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

        other_a = deepcopy(self)

        mean, covar = other_a._forward_gaussian(trans.state)
        noisey_v = other_a._noisey_vector()

        action = noisey_v @ covar + mean
        o = optim.Adam(other_a.parameters())

        jacobian = Tensor([])

        for elemenet in action:
            o.zero_grad()
            elemenet.backward(retain_graph=True)

            param_grad = Tensor([])
            for param in other_a.parameters():
                param_grad = torch.cat((param_grad, param.grad.flatten(start_dim=0)), dim = 0)
            
            jacobian = torch.cat((jacobian, param_grad.unsqueeze(dim=0)), dim = 0)





    
    def _forward_gaussian(self, x : Tensor) -> tuple[Tensor, Tensor]:
        """Will calculate the mean vector and the covar matrix"""

        output = self.layers(x)
        mean = output[:self.real_output_size]
        std = output[self.real_output_size:]
        covar = std.pow(2)
        covar = torch.diag(covar)

        return mean, covar
    
    #TODO make update parameters and sample from spherical guassian

    def _noisey_vector(self) -> Tensor:
        """Will create a vector epsilon sampled from a spherical guassian"""
        
        return self._s_dist.sample()






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
    
