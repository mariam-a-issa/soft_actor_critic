from random import randint

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import Tensor

import gym

import os
print(os.getcwd())

from discrete import NNAgent
from data_collection import MemoryBuffer, Transition

#Hyperparameters
#TODO Make these pass my command line

HIDDEN_LAYER_SIZE = 256
POLICY_LR = 3e-4
CRITIC_LR = 3e-4
DISCOUNT = .99
TAU = .005
ALPHA_SCALE = .98
TARGET_UPDATE = 8000
UPDATE_FREQUENCY = 4
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

def train(num_runs) -> None:
    """Will be the main training loop"""

    buffer = MemoryBuffer(BUFFER_SIZE, SAMPLE_SIZE)

    writer = SummaryWriter(LOG_DIR + f'/run{num_runs}')

    env = gym.make('CartPole-v1', render_mode = 'human')

    agent = NNAgent(
        4,
        2,
        HIDDEN_LAYER_SIZE,
        POLICY_LR,
        CRITIC_LR,
        DISCOUNT,
        TAU,
        ALPHA_SCALE,
        TARGET_UPDATE,
        UPDATE_FREQUENCY,
        writer
    )

    agent.to(_DEVICE)

    steps = 0
    num_games = 0
    total_return = 0
    episodic_reward = 0

    def get_action(s : Tensor) -> Tensor:
        if EXPLORE_STEPS <= steps:
            return agent(s)
        else:
            return torch.tensor(randint(0, 1))
        
    state = torch.tensor(env.reset()[0], device=_DEVICE, dtype=torch.float32)

    try:
        while MAX_STEPS > steps:
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
            
            if EXPLORE_STEPS <= steps:
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
        agent.save_actor(str(num_runs))
        env.close()

if __name__ == '__main__':
    for i in range(3):
        train(i)
