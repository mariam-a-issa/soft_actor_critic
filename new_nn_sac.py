from collections import namedtuple, deque
import random
from copy import deepcopy
import os
from pathlib import Path

import torch
from torch import nn, optim, tensor, Tensor
from torch.distributions import Normal
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import gym

LR = 3e-4
STEP_V = LR#1e-3
STEP_Q = LR#1e-3
STEP_ACTOR = LR
TAU = .005
GAMMA = .99
BUFFER_SIZE = 10 ** 6
SAMPLE_SIZE = 256
HIDDEN_LAYER_SIZE = 256
LOG_STD_MIN = -5
LOG_STD_MAX = 2
LEARNING_STEPS = 0

EPS = 1e-6 #So that we do not have a log(0) for the tanh squash of the actor

Transition = namedtuple('Transition', 
                        ['state', 
                         'action', 
                         'next_state', 
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
    
    def save(self, file_name : str='best_weights.pt') -> None:
        """Will save the model in the folder 'model' in the dir that the script was run in."""

        folder_name = type(self).__name__ + self.extra_info()

        model_folder_path = Path('./model/' + folder_name)
        file_dir = Path(os.path.join(model_folder_path, file_name))

        if not os.path.exists(file_dir.parent):
            os.makedirs(file_dir.parent)

        torch.save(self.state_dict(), file_dir)

    def extra_info(self) -> str:
        """Returns any extra information that describes the current NN"""
        return ''

class QModel(BaseNN):
    """Will be the model representing a q function.
    Its input should be the state and the action concatenated."""

    _next_id = 1

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._optim = optim.Adam(self.parameters(), STEP_Q)
        self._network_id = QModel._next_id
        QModel._next_id += 1
        self._num_updates = 1

    def update(self, loss : Tensor, num_games : int) -> None:
        """Find loss according to equations 7 and 8"""

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        writer.add_scalar(f'Q function {self._network_id} loss', loss, num_games)
        self._num_updates += 1

    def extra_info(self) -> str:
        return str(self._network_id)

class QFunction:
    """Will contain two q models that will be used as a single q function."""

    def __init__(self, q1 : QModel,  q2 :  QModel, *, alpha : float) -> None:
        self._q1 = q1
        self._q2 = q2
        self._alpha = alpha

    def __call__(self, x : Tensor) -> Tensor:
        """x should be the action concatenated to the state"""
        return torch.min(self._q1(x), self._q2(x))

    def update_parameters(self, trans : Transition, num_games : int) -> None:
        """Will update the parameters of the q models"""
        with torch.no_grad():
            input_tensor = torch.cat((trans.state, trans.action), dim=-1)

        q1 = self._q1(input_tensor)
        q2 = self._q2(input_tensor)

        #Find backup
        with torch.no_grad():
            new_action, new_log = self._actor(trans.next_state)
            new_in_tensor = torch.cat((trans.next_state, new_action), dim=-1)
            new_q = self(new_in_tensor)
            new_value =  new_q - self._alpha * new_log
            q_back = trans.reward + (1-trans.done) * GAMMA * new_value


        self._q1.update((1/2 * ((q1 - q_back) ** 2)).mean(), num_games)
        self._q2.update((1/2 * ((q2 - q_back) ** 2)).mean(), num_games)

    def upload_actor(self, actor : 'Actor') -> None:
        """Will upload the actor to the QFunction"""
        self._actor = actor

    def to(self, device) -> None:
        """Will move both of the q models to the device"""
        for model in self._list_q_funcs():
            model.to(device)
    
    def save(self, file_name : str='best_weights.pt') -> None:
        """Saves the best weights of each q model"""
        for model in self._list_q_funcs():
            model.save(file_name)

    def update_target(self, actual : 'QFunction') -> None:
        """Uses the actual to update self when self is a target"""
        with torch.no_grad():
            for param, target_param in zip(actual._q1.parameters(), self._q1.parameters()):
                target_param.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            for param, target_param in zip(actual._q2.parameters(), self._q2.parameters()):
                target_param.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def _list_q_funcs(self) -> list[QModel]:
        return [self._q1, self._q2]


class Actor(BaseNN):

    def __init__(self, input_size : int, output_size : int, *, q_function : QFunction, alpha : float) -> None:
        super().__init__(input_size, output_size)
        self._q_function = q_function

        #Split the last layer by removing it and having two seperate output layers
        #Allows batch training
        self.layers = self.layers[:-1]
        self._mean_lin = nn.Linear(self._hidden_size, output_size)
        self._covar_lin = nn.Linear(self._hidden_size, output_size)
        self._alpha = alpha

        self.optim = optim.Adam(self.parameters(), STEP_ACTOR) #, weight_decay=ACTOR_WEIGHT_DECAY) #weight decay is for l2 regularization which I think the github code uses I am not 100% though
        
        self._log2 = torch.log(tensor(2, device=_DEVICE, dtype=torch.float32))

        self._num_updates = 0

    def forward(self, x : Tensor) -> tuple[Tensor, Tensor]:
        """Will reuturn a tensor that represents the action for a single or batch of a state"""
        
        dist = self._dist(x)
        action = dist.rsample()

        return self._squash_output(action, dist)

    def update_parameters(self, trans : Transition, num_games : int) -> None:
        """Finds the loss using equation 12"""
        actions, log_probs = self(trans.state)

        input_tensor = torch.cat((trans.state, actions), dim=-1)
        q_value = self._q_function(input_tensor)
                                   
        error : Tensor = (self._alpha * log_probs - q_value)
        loss = error.mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        writer.add_scalar('Actor loss', loss, num_games)
        self._num_updates += 1

    def _squash_output(self, action : Tensor, dist : Normal) -> tuple[Tensor, Tensor]:
        """Will squash the action and the log_prob with tanh using equation 20
        Can be used on batches or on single values"""

        tan_action = torch.tanh(action)

        log_probs = dist.log_prob(action).sum(dim=-1)
        log_probs -= (2*(self._log2 - action - F.softplus(-2 * action))).sum(dim=-1)

        return tan_action, log_probs.unsqueeze(dim=-1)
    
    def _dist(self, x : Tensor) -> Normal:
        """Will create a distribution for either a single or batch of a state """
        
        net_out = self.layers(x)
        mean = self._mean_lin(net_out)
        std = self._covar_lin(net_out)
        std = torch.clamp(std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(std)

        return Normal(mean, std)


class MemoryBuffer:
    def __init__(self) -> None:
        self._memory = deque(maxlen=BUFFER_SIZE)

    def sample(self) -> Transition:
        """Will sample a batch of transitions from the replay buffer"""
        if len(self._memory) <= SAMPLE_SIZE:
            sample = self._memory #sample will be a list of transitions
        else:
            sample = random.sample(self._memory, SAMPLE_SIZE)

        state, action, next_state, reward, done = zip(*sample) #unpack list and create tuples of each data point in transition
        return Transition(state = torch.stack(state, dim = 0), #Each element of transition is the batch of values
                          action = torch.stack(action, dim = 0),
                          next_state = torch.stack(next_state, dim = 0),
                          reward = torch.stack(reward, dim = 0),
                          done = torch.stack(done, dim = 0))
    
    def add_data(self, trans : Transition) -> None:
        """Will add the data from the single transition into the buffer"""
        self._memory.append(trans)


def train(gm : gym.Env, len_state : int , len_output : int, * , reward_scale : float, max_game : int=None, max_steps : int=None, extra_save_info : str=None, log_dir :str=None) -> None:
    """Will train an agent with a continuous state space of dimensionality len_input and
    a continuous action space of dimensionality of len_output. It will train indefinitely until there
    is an exception (KeyboardInterrupt) or when the agent has been trained for a defined amount of max_game"""
    global writer
    if log_dir is not None:
        new_log_dir = f'runs/{log_dir}/run{extra_save_info}'
    else:
        new_log_dir = None     
    writer = SummaryWriter(log_dir=new_log_dir) 

    #Initialize all networks
    q_func = QFunction(QModel(len_state + len_output, 1),
                       QModel(len_state + len_output, 1),
                       alpha=1 / reward_scale)
    target_q_func = deepcopy(q_func)
    actor = Actor(len_state, len_output, q_function=target_q_func, alpha= 1 / reward_scale)
    q_func.upload_actor(actor)

    q_func.to(_DEVICE)
    actor.to(_DEVICE)
    
    replay_buffer = MemoryBuffer()
    num_games = 0
    steps = 0

    state = tensor(gm.reset()[0], device=_DEVICE, dtype=torch.float32)
    total_return = 0
    episodic_reward = 0

    def get_action(s : Tensor) -> tuple[Tensor, Tensor]:
        if LEARNING_STEPS < steps:
            return actor(s)
        else:
            return 2 * torch.rand(3) - 1, None

    try:
        while (max_game is None or max_game > num_games) and (max_steps is None or max_steps > steps):
            action, _ = get_action(state)
            next_state, reward, terminated, truncated, _ = gm.step(action.clone().detach().cpu().numpy())
            done = terminated or truncated
            total_return += reward
            episodic_reward += reward
            next_state = tensor(next_state, device=_DEVICE, dtype=torch.float32)
            trans = Transition( #states will be np arrays, actions will be tensors, the reward will be a float, and done will be a bool
                state,
                action,
                next_state,
                tensor([reward], device=_DEVICE, dtype=torch.float32),
                tensor([terminated], device=_DEVICE, dtype=torch.float32)
            )

            replay_buffer.add_data(trans)
            
            if LEARNING_STEPS < steps:
                batch = replay_buffer.sample()

                q_func.update_parameters(batch, steps)
                
                if num_games % 2:
                    actor.update_parameters(batch, steps)

                target_q_func.update_target(q_func)

            steps += 1

            if done:
                next_state = tensor(gm.reset()[0], device=_DEVICE, dtype=torch.float32)
                num_games += 1
                average_return = total_return / num_games
                writer.add_scalar('Average return', average_return, steps)
                writer.add_scalar('Episodic rerurn', episodic_reward, steps)
                episodic_reward = 0
            
            state = next_state
    finally:

        if log_dir is not None:
            file_name = Path(f'{log_dir}/best_weights')
        else:
            file_name = Path('best_weights')
        if extra_save_info is not None:
            file_name = Path(str(file_name) + extra_save_info)
        file_name = Path(str(file_name) + '.pt')

        actor.save(file_name)
        gm.close()


if __name__ == '__main__':
    env = gym.make('Hopper-v4', render_mode = 'human')
    train(env, 11, 3, reward_scale=5, log_dir='runs/test')
    
