from pathlib import Path

import gym
import torch

from nn_sac import Actor

def evaluate(env : gym.Env, input_size : int, output_size : int, *, path : str, max_games : int = None):
    """Will evaluate the model with the given input and output_size at the given path"""

    policy = Actor(input_size, output_size, q_function=None)
    policy.load_state_dict(torch.load(Path(path)))

    num_games = 0
    state = env.reset()[0]
    action, _ = policy(torch.tensor(state, dtype=torch.float32))


    try:
        while max_games is None or max_games < num_games:
            state, _, terminated, truncated, _ = env.step(action.clone().detach().cpu().numpy())
            
            if terminated or truncated:
                state = env.reset()[0]
                num_games += 1

            action, _ = policy(torch.tensor(state, dtype=torch.float32))

    finally:
        env.close()

if __name__ == '__main__':
    env = gym.make('Hopper-v4', render_mode='human')
    evaluate(env, 11, 3, path='C:\\Users\\ian\\Python\\hd_sac\\model\\Actor\\best_weights0.pt')