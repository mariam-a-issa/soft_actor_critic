from collections import namedtuple, deque
from math import pi
import random
import os

import torch
from torch import Tensor, tensor, nn, optim
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import gym

HYPER_DIM = 8192 #The dimensionality of the hypervectors
V_LR = .001
Q_LR = .001
A_LR  = .001
TAU = .005
DISCOUNT = .99
BUFFER_SIZE = 10 ** 6
SAMPLE_SIZE = 256

EPS = 1e-6

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'next_action', 'reward', 'done'])

if torch.cuda.is_available():
    device = f'cuda:{torch.cuda.current_device()}'
#elif torch.backends.mps.is_available():
#  device = 'mps'
else:
    device = 'cpu'
#device = 'cpu'

_DEVICE = torch.device(device)

class Buffer:

    def __init__(self, buffer_size, sample_size) -> None:
        self._buffer = deque(maxlen=buffer_size)
        self._sample_size = sample_size

    def remember(self, trans : Transition) -> None:
        """Will add the trans to the memory"""
        self._buffer.append(trans)

    def sample(self) -> Transition:
        """Will return a list of Transitions where the size n <= sample size"""
        if len(self._buffer) <= self._sample_size:
            sample = self._buffer #sample will be a list of transitions
        else:
            sample = random.sample(self._buffer, self._sample_size)

        state, action, next_state, next_action, reward, done = zip(*sample) #unpack list and create tuples of each data point in transition
        return Transition(state = torch.stack(state, dim = 0), #Each element of transition is the batch of values
                          action = torch.stack(action, dim = 0),
                          next_state = torch.stack(next_state, dim = 0),
                          next_action = torch.stack(next_action, dim = 0),
                          reward = torch.stack(reward, dim = 0),
                          done = torch.stack(done, dim = 0))


class EXPEncoder:
    """Represents the exponential encoder from the hdpg_actor_critic code but changed to only use a tensors from pytorch"""
    def __init__(self, d : int):
        """Will create an encoder that will work for vectors that have d dimensionality"""
        
        self._s_hdvec = torch.rand(HYPER_DIM, dtype=torch.float32, device=_DEVICE).unsqueeze(0)
        for _ in range(d - 1):
            self._s_hdvec = torch.cat((self._s_hdvec, torch.rand(HYPER_DIM, dtype=torch.float32, device=_DEVICE).unsqueeze(0)), dim = 0)
        
        self._bias = torch.rand(HYPER_DIM, dtype=torch.float32, device=_DEVICE) * 2 * pi
        self.d = d #Will be used by functions to know how to create their models

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the encoder hypervector. State needs the same dimensionality that was used to create the encoder"""
        return torch.exp(1j * (torch.matmul(state, self._s_hdvec) + self._bias))


class HDModel:
    """Represents a base class for a a hyperdimensional model that using the equation 5 in the hdpg paper"""
    def __init__(self) -> None:
        """Will create a model which"""
        temp_d = 32 #Could be a potential hyper parameter?
        self._m_hdvec = EXPEncoder(temp_d)(torch.rand(temp_d)) #Will need to find a way to initilze these parameters. Could use nn.Linear but that does not complex numbers which I think it needs?

    def __call__(self, x : Tensor) -> Tensor:
        """Will return a single value off of the vector x which should have a dimensioanilty of HYPER_DIM"""
        return torch.real((torch.dot(torch.conj(x), self._m_hdvec)) / HYPER_DIM)
    
    def params(self) -> Tensor:
        """Will return the parameters of the network which is just the hypervector"""
        return self._m_hdvec
    
    def update_params(self, u : Tensor) -> None:
        """Will set the vector u as the new params of the network"""
        self._m_hdvec = u
        


class QModel(HDModel):

    _next_id = 1

    def __init__(self, state_en : EXPEncoder, action_en : EXPEncoder, target_v : 'TargetValueFunction') -> None:
        super().__init__()
        self._t_v = target_v
        self._state_en = state_en
        self._action_en = action_en
        self._model_id = QModel._next_id
        QModel._next_id += 1
        self._num_updates = 0

    def __call__(self, state : Tensor, action : Tensor) -> Tensor:
        """Will create a hypervector that represents both the state and the action and then call the model bassed off of this.
        The state and action should not be encoded and have their respective dimensions"""
        if len(state.shape) == 1:
            state_action_hdvec = torch.add(self._state_en(state), self._action_en(action))
            return super().__call__(state_action_hdvec)
        
        #Make a vector of the q_values
        state_action_hdvec = torch.add(self._state_en(state[0]), self._action_en(action[0]))
        q_vals = super().__call__(state_action_hdvec).unsqueeze(dim=0)

        for s, a in zip(state, action):
            sa_hdvec = torch.add(self._state_en(s), self._action_en(a))
            q_vals = torch.cat((q_vals, super().__call__(sa_hdvec).unsqueeze(dim=0)), dim=0)

        return q_vals

    
    def update(self, buffer : Buffer) -> Tensor:
        """Will update the q functions using equation 9 and algorithim 1 of the SAC paper"""

        sample = buffer.sample()
        q_vals = self(sample.state, sample.action)
        v_vals = self._t_v(sample.next_state)

        average_loss = 0
        total_losses = len(sample)
        for r, q, v, d in zip(sample.reward, q_vals, v_vals, sample.done):
            loss = q - (r + (1-d) * DISCOUNT * v)
            new_params = self.params() + Q_LR * (loss) * self.params()
            self.update_params(new_params)
            average_loss += loss

        average_loss /= total_losses
        self._num_updates +=1

        writer.add_scalar(f'Q func loss {self._model_id}', average_loss, self._num_updates)


class QFunction:
    def __init__(self, state_en : EXPEncoder, action_en : EXPEncoder, target_v : 'TargetValueFunction') -> None:
        self._q1 = QModel(state_en, action_en, target_v)
        self._q2 = QModel(state_en, action_en, target_v)

    def __call__(self, state : Tensor, action : Tensor) -> Tensor:
        """Will return the q function that has the minimum value according to the SAC paper"""
        return torch.min(self._q1(state, action), self._q2(state, action))
    
    def update(self, trans : Transition) -> None:
        """Will update the both of the q models in the q function"""
        self._q1.update(trans)
        self._q2.update(trans)

class ValueFunction(HDModel):
    def __init__(self, encoder : EXPEncoder, q_func : QFunction, actor : 'Actor') ->None:
        super().__init__()
        self._encoder = encoder
        self._q_func = q_func
        self._actor = actor
        self._num_updates = 0

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the value of the state where state is a vector of state dimensioanlity"""
        if len(state.shape) == 1:
            return super().__call__(self._encoder(state))
        
        #Make a vector of the v_values
        v_vals = super().__call__(self._encoder(state[0])).unsqueeze(dim = 0)

        for s in state[1:]:
            v_vals = torch.cat((v_vals, super().__call__(self._encoder(s)).unsqueeze(dim=0)), dim=0)

        return v_vals

    def update(self, buffer : Buffer) -> None:
        """Updates the value function using equation 6 and algorithim 1 on the SAC paper"""
        sample = buffer.sample()
        actions, log_probs = self._actor(sample.state)
        q_vals = self._q_func(sample.state, actions)
        v_vals = self(sample.state)

        average_loss = 0
        num_losses = len(sample)
        for lp, q, v in zip(log_probs, q_vals, v_vals):
            loss = v - (q - lp)
            new_params = self.params() + V_LR * (loss) * self.params()
            self.update_params(new_params)
            average_loss += loss

        average_loss /= num_losses
        self._num_updates += 1

        writer.add_scalar('V func loss', average_loss, self._num_updates)

        


class TargetValueFunction(ValueFunction):
    def __init__(self, encoder : EXPEncoder):
        super().__init__(encoder, None, None)
        
    def upload_v_func(self, v_func : ValueFunction):
        self._v_func = v_func

    def update(self, buffer : Buffer) -> None:
        """Will update the function according to equation in algorithm 1 of the SAC paper"""
        self.update_params(TAU * self._v_func.params() + (1 - TAU) * self.params())

class RBFEncoder:
    def __init__(self, in_size: int, out_size: int = HYPER_DIM):
        self._in_size = in_size
        self._out_size = out_size

        # TODO: may need change
        self._s_hdvec = torch.randn(in_size, out_size, dtype=torch.float32, device=_DEVICE) / in_size #Why normalize with in_size
        self._bias = 2 * pi * torch.randn(out_size, dtype=torch.float32, device=_DEVICE)
  

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: need to check dimension of x
        x = x @ self._s_hdvec + self._bias
        x = torch.cos(x)
        return x

class Actor(nn.Module):

    def __init__(self, encoder : RBFEncoder, out_size : int, q_func : QFunction, dim_size = HYPER_DIM):
        super().__init__()
        self._encoder = encoder
        self._mean = nn.Linear(dim_size, out_size, bias=False)
        self._covar = nn.Linear(dim_size, out_size, bias=False)
        self._q_func = q_func
        self._optim = optim.SGD(self.parameters(), lr = A_LR)
        self._num_updates = 0

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Will reuturn a tensor that represents the action for a single or batch of a state"""
        
        dist = self._dist(self._encoder(state))
        action = dist.sample()

        return self._squash_output(action, dist)

    def update(self, buffer : Buffer) -> None:
        """Finds the loss using equation 12"""
        sample = buffer.sample()
        dist = self._dist(self._encoder(sample.state))
        action = dist.rsample()
        action, log_probs = self._squash_output(action, dist)
        
        with torch.no_grad():
            q_value = self._q_func(sample.state, action)

        loss = (log_probs - q_value).mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        self._num_updates += 1

        writer.add_scalar('Actor loss', loss, self._num_updates)

    def save(self, file_name : str='best_weights.pt') -> None:
        """Will save the model in the folder 'model' in the dir that the script was run in."""

        folder_name = type(self).__name__

        model_folder_path = './model/' + folder_name

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_dir = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_dir)
        

    def _squash_output(self, action : Tensor, dist : MultivariateNormal) -> tuple[Tensor, Tensor]:
        """Will squash the action and the log_prob with tanh using equation 20
        Can be used on batches or on single values"""

        tan_action = torch.tanh(action)
        log_probs : Tensor = (dist.log_prob(action) - torch.sum(torch.log(1 - torch.tanh(action) ** 2 + EPS), dim = -1)).unsqueeze(-1)

        return tan_action, log_probs

    def _dist(self, x : Tensor) -> MultivariateNormal:
        """Will create a distribution for either a single or batch of a state """

        mean = self._mean(x)
        covar = self._covar(x)
        covar = torch.clamp(covar, EPS, 1)
    
        if len(covar.shape) == 2: #When dealing with batches
            covar_m = torch.diag(covar[0]).unsqueeze(dim = 0)
            for cv in covar[1:]:
                covar_m = torch.cat((covar_m, torch.diag(cv).unsqueeze(dim = 0)), dim = 0)
        else:
            covar_m = torch.diag(covar)

        return MultivariateNormal(mean, covar_m)
    

def train(gm : gym.Env, len_state : int , len_output : int, * , reward_scale : float, max_game : int=None, max_steps : int=None, extra_save_info : str=None, log_dir :str=None) -> None:
    """Will train an agent with a continuous state space of dimensionality len_input and
    a continuous action space of dimensionality of len_output. It will train indefinitely until there
    is an exception (KeyboardInterrupt) or when the agent has been trained for a defined amount of max_game"""

    global writer #Should be fixed later
    writer = SummaryWriter(log_dir=log_dir)

    c_state_encoder = EXPEncoder(len_state)
    action_encoder = EXPEncoder(len_output)
    a_state_encoder = RBFEncoder(len_state)

    #Initialize all networks
    target_v = TargetValueFunction(c_state_encoder)
    q_func = QFunction(c_state_encoder, action_encoder, target_v)
    actor = Actor(a_state_encoder, len_output, q_func)
    v_func = ValueFunction(c_state_encoder, q_func, actor)
    target_v.upload_v_func(v_func)


    #list_networks = [q_func, actor]
    list_networks = [v_func, q_func, actor, target_v] #This list should be used if training actor
    
    buffer = Buffer(BUFFER_SIZE, SAMPLE_SIZE)
    num_games = 0
    episodes = 0

    state = gm.reset()[0]
    action, _ = actor(tensor(state, dtype=torch.float32, device=_DEVICE))

    total_return = 0

    try:
        while (max_game is None or max_game > num_games) and (max_steps is None or max_steps > episodes):
            unscaled_action = action.clone().detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = gm.step(unscaled_action)
            done = terminated or truncated
            total_return += reward
            reward *= reward_scale
            next_state = tensor(next_state, dtype=torch.float32, device=_DEVICE)
            next_action, _ = actor(next_state)

            trans = Transition( #states will be np arrays, actions will be tensors, the reward will be a float, and done will be a bool
                tensor(state, dtype=torch.float32, device=_DEVICE),
                action,
                tensor(next_state, dtype=torch.float32, device=_DEVICE),
                next_action,
                tensor([reward], dtype=torch.float32, device=_DEVICE),
                tensor([done], dtype=torch.float32, device=_DEVICE)
            )

            buffer.remember(trans)
            
            for model in list_networks:
                model.update(buffer)

            episodes += 1

            if done:
                next_state = gm.reset()[0]
                next_action, _ = actor(tensor(next_state, dtype=torch.float32, device=_DEVICE))
                num_games += 1
                average_return = total_return / num_games
                writer.add_scalar('Average return', average_return, episodes)
            
            state = next_state
            action = next_action
    finally:

        file_name = 'best_weights'
        if extra_save_info is not None:
            file_name += extra_save_info
        file_name += '.pt'

        actor.save(file_name)
        gm.close()

if __name__ == '__main__':
    env = gym.make('Hopper-v4', render_mode = 'human')
    train(env, 11, 3, reward_scale=5.0)







