import argparse
from pathlib import Path

import gym

from continuous.new_nn_sac import train

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--log_dir')
    parser.add_argument('--max_steps')
    args = parser.parse_args()

    log_dir = args.log_dir
    max_steps = (args.max_steps)

    assert log_dir is not None, 'Run with --log_dir argument specifiying what dir to save into'
    
    if max_steps is None:
        max_steps = 10**6
    else:
        max_steps = float(max_steps)

    for i in range(3):
        env = gym.make('Hopper-v4')
        train(env, 11, 3, reward_scale=5.0, max_steps=max_steps, extra_save_info=str(i), log_dir=log_dir)



if __name__ == '__main__':
    main()