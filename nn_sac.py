from collections import namedtuple, deque
import random
from copy import deepcopy

import torch
from torch import nn, optim, tensor, Tensor
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

import gym

LR = 3 * (10 ** -4)
STEP_V = LR
STEP_Q = LR
STEP_ACTOR = LR
TAU = .005
GAMMA = .99
BUFFER_SIZE = 10 ** 6
SAMPLE_SIZE = 256
HIDDEN_LAYER_SIZE = 256
ACTOR_WEIGHT_DECAY = 1e-3 #Found regularization in code

EPS = 1e-6 #So that we do not have a log(0) for the tanh squash of the actor

Transition = namedtuple('Transition', 
                        ['state', 
                         'action', 
                         'next_state', 
                         'next_action', 
                         'reward', 
                         'done']) #If bad performance just switch to a tensor

if torch.cuda.is_available():
    device = f'cuda:{torch.cuda.current_device()}'
#elif torch.backends.mps.is_available():
#  device = 'mps'
else:
    device = 'cpu'
#device = 'cpu'

_DEVICE = torch.device(device)

torch.set_default_device(_DEVICE)

#Throughout code I used n x data_dims where n is the batch size. If not using a batch n = 1

class BaseNN(nn.Module):
    """Base class for constructing NNs"""

    def __init__(self, input_size : int,  output_size : int, hidden_size : int = HIDDEN_LAYER_SIZE) -> None:
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

    def __init__(self, *args, q_function : 'QFunction', actor : 'Actor') -> None:
        super().__init__(*args)
        self._optim = optim.Adam(self.parameters(), STEP_V)
        self._actor = actor
        self._q_func = q_function

        self._num_updates = 1 #For logging

    def update_parameters(self, trans : Transition) -> None:
        """Use equation 5 to update"""
        actions, log_prob = self._actor(trans.state)
        input_tensor = torch.cat((trans.state, actions), dim = 1)

        #For a regularization term which was included Haarnoja github implementation
        data_s = actions.shape[-1]
        num_batches = actions.shape[0]
        policy_prior = MultivariateNormal(
            loc=torch.zeros(data_s).unsqueeze(0).repeat(num_batches, 1),
            covariance_matrix=torch.eye(data_s).unsqueeze(0).repeat(num_batches, 1, 1))
        policy_prior_log_probs = policy_prior.log_prob(actions)

        error : Tensor = (self(trans.state) - (self._q_func(input_tensor) - log_prob + policy_prior_log_probs)) ** 2
        loss = 1/2 * error.mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        writer.add_scalar('Value loss', loss, self._num_updates)
        self._num_updates += 1



class TargetValueFunction:
    """Target target value function that contains method to update parameters not using gradient according to pseudocode""" #Is the pseudocode representation the exponentially moving average?

    def __init__(self) -> None:
        self._v_tar = None
        self._v_func = None

    def upload_v_func(self, v_func : ValueFunction) -> None:
        """Uploads the v function after the target is made"""
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
    Its input should be the state and the action concatenated."""

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

        error : Tensor = (self(input_tensor) - (trans.reward + GAMMA  * (1- trans.done) * self._v_func(trans.next_state))) ** 2
        loss = 1/2 * error.mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        writer.add_scalar(f'Q function {self._network_id} loss', loss, self._num_updates)
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

    def to(self, device) -> None:
        """Will move both of the q models to the device"""
        for model in self._list_q_funcs():
            model.to(device)

    def _list_q_funcs(self) -> list[QModel]:
        return [self._q1, self._q2]


class Actor(BaseNN):

    def __init__(self, input_size : int, output_size : int, *, q_function : QFunction) -> None:
        super().__init__(input_size, output_size)
        self._q_function = q_function

        #Split the last layer by removing it and having two seperate output layers
        #Allows batch training
        self.layers = self.layers[:-1]
        self._mean_lin = nn.Linear(self._hidden_size, output_size)
        self._covar_lin = nn.Linear(self._hidden_size, output_size)

        def all_params():
            yield from self.layers.parameters()
            yield from self._mean_lin.parameters()
            yield from self._covar_lin.parameters()

        self._optim = optim.Adam(all_params(), STEP_ACTOR, weight_decay=ACTOR_WEIGHT_DECAY)

        self._num_updates = 0

    def forward(self, x : Tensor) -> tuple[Tensor, Tensor]:
        """Will reuturn a tensor that represents the action for a single or batch of a state"""
        
        dist = self._dist(x)
        action = dist.sample()

        return self._squash_output(action, dist)

    def update_parameters(self, trans : Transition) -> None:
        """Updates the parameters using equation 12"""

        dist = self._dist(trans.state)
        actions = dist.rsample()
        actions, log_probs = self._squash_output(actions, dist)
        input_tensor = torch.cat((trans.state, actions), dim=1)
        error : Tensor = log_probs - self._q_function(input_tensor)
        loss = error.mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        writer.add_scalar('Actor loss', loss, self._num_updates)
        self._num_updates += 1

    def _squash_output(self, action : Tensor, dist : MultivariateNormal) -> tuple[Tensor, Tensor]:
        """Will squash the action and the log_prob with tanh using equation 20
        Can be used on batches or on single values"""

        tan_action = torch.tanh(action)
        log_probs : Tensor = (dist.log_prob(action) - torch.sum(torch.log(1 - torch.tanh(action) ** 2 + EPS), dim = -1)).unsqueeze(-1)

        return tan_action, log_probs
        #return action, dist.log_prob(action).unsqueeze(dim = -1)

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


class MemoryBuffer:
    def __init__(self) -> None:
        self._memory = deque(maxlen=BUFFER_SIZE)

    def sample(self) -> Transition:
        """Will sample a batch of transitions from the replay buffer"""
        if len(self._memory) <= SAMPLE_SIZE:
            sample = self._memory #sample will be a list of transitions
        else:
            sample = random.sample(self._memory, SAMPLE_SIZE)

        state, action, next_state, next_action, reward, done = zip(*sample) #unpack list and create tuples of each data point in transition
        return Transition(state = torch.stack(state, dim = 0), 
                          action = torch.stack(action, dim = 0),
                          next_state = torch.stack(next_state, dim = 0),
                          next_action = torch.stack(next_action, dim = 0),
                          reward = torch.stack(reward, dim = 0),
                          done = torch.stack(done, dim = 0))
    
    def add_data(self, trans : Transition) -> None:
        """Will add the data from the single transition into the buffer"""
        self._memory.append(trans)

def train(gm : gym.Env, len_state : int , len_output : int, * , reward_scale : float, max_game : int = None, max_steps : int = None) -> None:
    """Will train an agent with a continuous state space of dimensionality len_input and
    a continuous action space of dimensionality of len_output. It will train indefinitely until there
    is an exception (KeyboardInterrupt) or when the agent has been trained for a defined amount of max_game"""

    global writer #Should be fixed later
    writer = SummaryWriter() 

    #Initialize all networks
    target_v = TargetValueFunction()
    q_func = QFunction(QModel(len_state + len_output, 1, target_v_func=target_v),
                       QModel(len_state + len_output, 1, target_v_func=target_v))
    actor = Actor(len_state, len_output, q_function=q_func)
    v_func = ValueFunction(len_state, 1, q_function=q_func, actor=actor)
    target_v.upload_v_func(v_func)

    q_func.to(_DEVICE)
    actor.to(_DEVICE)
    v_func.to(_DEVICE)

    list_networks = [v_func, q_func, actor, target_v]
    #list_networks = [v_func, q_func, actor, target_v] This list should be used if training actor
    
    replay_buffer = MemoryBuffer()
    num_games = 0
    episodes = 0

    state = gm.reset()[0]
    action, _ = actor(tensor(state, device=_DEVICE, dtype=torch.float32)) #can use random action torch.FloatTensor(1).uniform_(-2.0, 2.0)

    total_return = 0

    try:
        while (max_game is None or max_game > num_games) and (max_steps is None or max_steps > episodes):
            next_state, reward, terminated, truncated, _ = gm.step(action.clone().detach().cpu().numpy())
            done = terminated or truncated
            total_return += reward
            reward *= reward_scale
            next_state = tensor(next_state, device=_DEVICE, dtype=torch.float32)
            next_action, _ = actor(next_state)

            trans = Transition( #states will be np arrays, actions will be tensors, the reward will be a float, and done will be a bool
                tensor(state, device=_DEVICE, dtype=torch.float32),
                action,
                tensor(next_state, device=_DEVICE, dtype=torch.float32),
                next_action,
                tensor([reward], device=_DEVICE, dtype=torch.float32),
                tensor([done], device=_DEVICE, dtype=torch.float32)
            )

            replay_buffer.add_data(trans)
            
            batch = replay_buffer.sample()
            for net in list_networks:
                net.update_parameters(batch)

            episodes += 1

            if done:
                state = gm.reset()[0]
                action, _ = actor(tensor(state, device=_DEVICE, dtype=torch.float32)) #can use random action torch.FloatTensor(1).uniform_(-2.0, 2.0)
                num_games += 1
                average_return = total_return / num_games
                writer.add_scalar('Average return', average_return, episodes)
            
            state = next_state
            action = next_action
    finally:
        gm.close()


if __name__ == '__main__':
    env = gym.make('Hopper-v4', render_mode = 'human')
    train(env, 11, 3, reward_scale=5.0)
    
