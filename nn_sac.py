import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.nn import MSELoss
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
        """Using batchs x should be N x D where N is the number of batches"""
        return self.layers(x)
    

class ValueFunction(BaseNN):
    """Will be target function that updates parameters according to equation 6"""

    def __init__(self, *args, q_function : 'QFunction', actor : 'Actor') -> None:
        super().__init__(*args)
        self._optim = optim.Adam(self.parameters(), STEP_V)
        self._actor = actor
        self._q_func = q_function

        self._num_updates = 1 #For logging

    def update_parameters(self, trans : Transition) -> None:
        actions, log_prob = self._actor(trans.state)
        input_tensor = torch.cat((trans.state, actions), dim = 1)
        error : Tensor = 1/2 * ((self(trans.action) - (self._q_func(input_tensor) - log_prob)) ** 2)
        ex_error = error.mean()

        self._optim.zero_grad()
        ex_error.backward()
        self._optim.step()

        writer.add_scalar('Value error', ex_error, self._num_updates)
        self._num_updates += 1



class TargetValueFunction:
    """Target target value function that contains method to update parameters not using gradient according to pseudocode""" #Is the pseudocode representation the exponentially moving average?

    def __init__(self) -> None:
        self._v_tar = None
        self._v_func = None

    def upload_v_func(self, v_func : ValueFunction) -> None:
        self._v_tar = deepcopy(v_func)
        self._v_func = v_func

    def update_parameters(self, trans : Transition) -> None:
        """Update parameters according to algorithim 1"""
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
        """Update parameters according to equations 7 and 8"""
        input_tensor = torch.cat((trans.state, trans.action), dim = 1)
        error : Tensor = 1/2 * ((self(input_tensor) - (trans.reward + self._v_func(trans.next_state))) ** 2)
        ex_error = error.mean()

        self._optim.zero_grad
        ex_error.backward()
        self._optim.step()

        writer.add_scalar(f'Q function {self._network_id} error', ex_error, self._num_updates)
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
        """Will update both q models of the q functions"""
        for q in self._list_q_funcs():
            q.update_parameters(trans)

    def _list_q_funcs(self) -> list[QModel]:
        return [self._q1, self._q2]


class Actor(BaseNN):
    """Actor updated according to equation 13"""

    def __init__(self, input_size : int, output_size : int, *, q_function : QFunction) -> None:
        super().__init__(input_size, output_size)
        self._q_function = q_function

        #Split the last layer by removing it and having two seperate output layers
        #Allows batch training
        self.layers = self.layers[:-1]
        self._mean_lin = nn.Linear(self._hidden_size, output_size)
        self._covar_lin = nn.Linear(self._hidden_size, output_size)
        self._optim = optim.Adam(self.layers.parameters() + self._mean_lin() + self._covar_lin.parameters(),
                                 STEP_ACTOR)
    
    def forward(self, x : Tensor) -> Tensor:
        """Will reuturn a tensor that represents the action for a single or batch of a state"""
        dist = self._dist(x)
        action = dist.sample()
        return action, dist.log_prob(action)

    
    def update_parameters(self, trans : Transition) -> None:
        """Updates the parameters using equation 12"""

        dist = self._dist(trans.state)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        loss : Tensor = log_probs - self._q_function(torch.cat((trans.state, actions), dim=1)) #Concate in dimension where the vectors representing state and action are stored
        
        self._optim.zero_grad()
        loss.mean().backward()
        self._optim.step()

    def _dist(self, x : Tensor) -> MultivariateNormal:
        """Will create a distribution for either a single or batch of a state """

        mean = self._mean_lin(self.layers(x))
        covar = self._covar_lin(self.layers(x))
        covar = covar.pow(2)

        if len(covar.shape) == 2: #When dealing with batches
            covar_m = torch.diag(covar[0]).unsqueeze(dim = 0)
            for cv in covar[1:]:
                covar_m = torch.cat((covar_m, torch.diag(cv).unsqueeze(dim = 0)), dim = 0)
        else:
            covar_m = torch.diag(covar)

        return MultivariateNormal(mean, covar_m)
    
    
    def _forward_gaussian(self, x : Tensor) -> tuple[Tensor, Tensor]:
        """Will calculate the mean vector and the covar matrix"""

    def _forward_gaussian(self, x : Tensor) -> tuple[Tensor, Tensor]:
        """Will calculate the mean vector and the covar matrix"""


        output = self.layers(x)
        mean = output[:self.real_output_size]
        std = output[self.real_output_size:]
        covar = std.pow(2)
        covar = torch.diag(covar)
    
        output = self.layers(x)
        mean = output[:self.real_output_size]
        std = output[self.real_output_size:]
        covar = std.pow(2)
        covar = torch.diag(covar)
    
    #TODO make update parameters and sample from spherical guassian
        
    #TODO make update parameters and sample from spherical guassian
        
        return self._s_dist.sample()

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
    episodes = 0

    state = gm.reset()
    state = Tensor(state[0])
    action, _ = actor.action_log(state) #can use random action torch.FloatTensor(1).uniform_(-2.0, 2.0)
    
    try:
        while max_game is None or max_game > num_games:
            next_state, reward, done, _, _ = gm.step(action.detach().numpy())
            next_state = Tensor(next_state)
            next_action, _ = actor(next_state)

            trans = Transition(state, action, next_state, next_action, reward)

            for net in list_networks:
                net.update_parameters(trans)

            episodes += 1

            if done or episodes > 500:
                state = gm.reset()
                action, _ = actor(state) #can use random action torch.FloatTensor(1).uniform_(-2.0, 2.0)
                num_games += 1
                episodes = 0
            
            state = next_state
            action = next_action
    finally:
        gm.close()

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3', render_mode = 'human')
    train(env, 24, 4)
    
