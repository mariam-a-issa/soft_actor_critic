from random import randint

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import Tensor

import gym

from discrete import NNAgent
from data_collection import MemoryBuffer, Transition

#Hyperparameters
#TODO Make these pass my command line

HIDDEN_LAYER_SIZE = 256
POLICY_LR = 3e-4
CRITIC_LR = 3e-4
DISCOUNT = .99
TAU = .005
ALPHA_SCALE = .89
TARGET_UPDATE = 1
UPDATE_FREQUENCY = 1
EXPLORE_STEPS = 0
BUFFER_SIZE = 10 ** 6
SAMPLE_SIZE = 64

LOG_DIR = './runs/first_test'

MAX_STEPS = 1e5

if torch.cuda.is_available():
    device = f'cuda:{torch.cuda.current_device()}'
else:
    device = 'cpu'

_DEVICE = torch.device(device)

torch.set_default_device(_DEVICE)

def train(
        extra_info : str = '', *,
        log_dir : str = LOG_DIR,
        hidden_size : int = HIDDEN_LAYER_SIZE,
        policy_lr : float = POLICY_LR,
        critic_lr : float= CRITIC_LR,
        discount : float = DISCOUNT,
        tau : float = TAU,
        alpha_scale :float = ALPHA_SCALE,
        target_update : int = TARGET_UPDATE,
        update_frequency : int = UPDATE_FREQUENCY,
        explore_steps : int = EXPLORE_STEPS ,
        buffer_size : int = BUFFER_SIZE,
        sample_size : int = SAMPLE_SIZE,
        max_steps : int = MAX_STEPS) -> None:
    """Will be the main training loop"""

    buffer = MemoryBuffer(buffer_size, sample_size)

    writer = SummaryWriter(log_dir + f'/run{extra_info}')

    env = gym.make('CartPole-v1')

    agent = NNAgent(
        4,
        2,
        hidden_size,
        policy_lr,
        critic_lr,
        discount,
        tau,
        alpha_scale,
        target_update,
        update_frequency,
        writer
    )

    agent.to(_DEVICE)

    steps = 0
    num_games = 0
    total_return = 0
    episodic_reward = 0

    def get_action(s : Tensor) -> Tensor:
        if explore_steps <= steps:
            return agent(s)
        else:
            return torch.tensor(randint(0, 1))
        
    state = torch.tensor(env.reset()[0], device=_DEVICE, dtype=torch.float32)

    try:
        while max_steps > steps:
            action = get_action(state).unsqueeze(dim = 0)
            next_state, reward, terminated, truncated, _ = env.step(action.clone().detach().cpu().item())
            done = terminated or truncated
            total_return += reward
            episodic_reward += reward
            next_state = torch.tensor(next_state, device=_DEVICE, dtype=torch.float32)
            trans = Transition( #states will be np arrays, actions will be tensors, the reward will be a float, and done will be a bool
                state,
                action,
                next_state,
                torch.tensor([reward], device=_DEVICE, dtype=torch.float32),
                torch.tensor([terminated], device=_DEVICE, dtype=torch.float32)
            )

            buffer.add_data(trans)
            
            if explore_steps <= steps:
                agent.update(buffer.sample(), steps)

            steps += 1

            if done:
                next_state = torch.tensor(env.reset()[0], device=_DEVICE, dtype=torch.float32)
                num_games += 1
                average_return = total_return / num_games
                writer.add_scalar('Average return', average_return, steps)
                writer.add_scalar('Episodic rerurn', episodic_reward, steps)
                episodic_reward = 0
            
            state = next_state

    finally:
        agent.save_actor(extra_info)
        env.close()

if __name__ == '__main__':
    for i in range(3):
        train(i)
