import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.distributions import MultivariateNormal
from collections import namedtuple
from copy import deepcopy
import gym
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

LR = 3 * (10 ** -4)
STEP_V = LR
STEP_Q = LR
STEP_ACTOR = LR
TAU = .005
GAMMA = .99

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'next_action', 'reward']) #If bad performance just switch to a tensor


#TODO How to make it a batch for the updates?


class BaseNN(nn.Module):
    """Base class for constructing NNs"""

    def __init__(self, input_size : int,  output_size : int, hidden_size : int = 256) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
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
        self._optim = optim.Adam(self.parameters(), STEP_V)
        self._actor = actor
        self._q_function = q_function

        self._num_updates = 1 #For logging

    def update_parameters(self, trans : Transition) -> None:
        action, log_prob = self._actor(trans.state)
        error = float(self(trans.state) - self._q_function(torch.cat((trans.state, action), dim = 0)) + log_prob)
        
        val = self(trans.state) * error
        self._optim.zero_grad()
        val.backward()

        self._optim.step()

        writer.add_scalar('Value error', error, self._num_updates)
        self._num_updates += 1



class TargetValueFunction:
    """Target target value function that contains method to update parameters not using gradient according to pseudocode""" #Is the pseudocode representation the exponentially moving average?

    def upload_v_func(self, v_func : ValueFunction) -> None:
        self._v_tar = deepcopy(v_func)
        self._v_func = v_func

    def update_parameters(self, trans : Transition) -> None:
        with torch.no_grad():
            for t_param, v_param in zip(self._v_tar.parameters(), self._v_func.parameters()):
                new_value = TAU * v_param + (1 - TAU) * t_param
                t_param.copy_(new_value)


    def __call__(self, state : Tensor) -> Tensor:
        return self._v_tar(state)


class QModel(BaseNN):
    """Will be the model representing a q function.
    Its input should be the state and the action concatenated.
    Updated according to the equation 9"""

    _next_id = 1

    def __init__(self, *args, target_v_func : TargetValueFunction) -> None:
        super().__init__(*args)
        self._v_func = target_v_func
        self._optim = optim.Adam(self.parameters(), STEP_Q)

        self._network_id = QModel._next_id
        QModel._next_id += 1
        self._num_updates = 1
    
    def update_parameters(self, trans : Transition) -> None:
        input_tensor = torch.cat((trans.state, trans.action), dim = 0)
        error = float(self(input_tensor)) - trans.reward - GAMMA * self._v_func(trans.next_state)

        val = self(input_tensor) * error
        self._optim.zero_grad()
        val.backward()
        self._optim.step()

        writer.add_scalar(f'Q function {self._network_id} error', error, self._num_updates)
        self._num_updates += 1




class QFunction:
    """Will contain two q models that will be used as a single q function."""

    def __init__(self, q1 : QModel,  q2 :  QModel) -> None:
        self._q1 = q1
        self._q2 = q2

    def __call__(self, x : Tensor) -> Tensor:
        """x should be the action concatenated to the state"""
        return torch.min(self._q1(x), self._q2(x))

    def update_parameters(self, trans : Transition) -> None:
        for q in self._list_q_funcs():
            q.update_parameters(trans)

    def _list_q_funcs(self) -> list[QModel]:
        return [self._q1, self._q2]


class Actor(BaseNN):
    """Actor updated according to equation 13"""

    def __init__(self, input_size : int, output_size : int, *, q_function : QFunction) -> None:
        super().__init__(input_size, output_size * 2) #Needs to be twice as we need to get mean and covar vectors
        self._q_function = q_function
        self.real_output_size = output_size

        #May need to pick the random numbers differently
        mean = torch.randn(self.real_output_size) 
        variance = float(torch.rand(1)) 

        covar = variance * torch.eye(self.real_output_size)
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
        noisy_v = other_a._noisy_vector()

        action = noisy_v @ covar + mean
        o = optim.Adam(other_a.parameters())

        jacobian = Tensor([])

        for element in action:
            o.zero_grad()
            element.backward(retain_graph=True)

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

    def _noisy_vector(self) -> Tensor:
        """Will create a vector epsilon sampled from a spherical guassian"""
        
        return self._s_dist.sample()






def train(gm : gym.Env, len_state : int , len_output : int, * , max_game : int = None) -> None:
    """Will train an agent with a continuous state space of dimensionality len_input and
    a continuous action space of dimensionality of len_output. It will train indefinitely until there
    is an exception (KeyboardInterrupt) or when the agent has been trained for a defined amount of max_game"""

    #Initialize all networks
    target_v = TargetValueFunction()
    q_func = QFunction(QModel(len_state + len_output, 1, target_v_func=target_v),
                       QModel(len_state + len_output, 1, target_v_func=target_v))
    actor = Actor(len_state, len_output, q_function=q_func)
    v_func = ValueFunction(len_state, 1, q_function=q_func, actor=actor)
    target_v.upload_v_func(v_func)

    #List of 
    list_networks = [v_func, q_func, target_v]
    #list_networks = [v_func, q_func, actor, target_v] This list should be used if training actor

    num_games = 0

    state = gm.reset()
    state = Tensor(state[0])
    action, _ = actor(state) #can use random action torch.FloatTensor(1).uniform_(-2.0, 2.0)
    
    try:
        while max_game == None or max_game > num_games:
            print(action)
            next_state, reward, done, _, _ = gm.step(action.detach().numpy())
            next_state = Tensor(next_state)
            next_action, _ = actor(next_state)

            trans = Transition(state, action, next_state, next_action, reward)

            for net in list_networks:
                net.update_parameters(trans)

            if done:
                state = gm.reset()
                action, _ = actor(state) #can use random action torch.FloatTensor(1).uniform_(-2.0, 2.0)
                num_games += 1

            state = next_state
            action = next_action
    finally:
        gm.close()

if __name__ == '__main__':
    env = gym.make('Pendulum-v1', render_mode='human')
    train(env, 3, 1)
    
