import gym

from nn_sac import train

for i in range(2):
    env = gym.make('Hopper-v4')
    train(env, 11, 3, reward_scale=5.0, max_steps=10**6, extra_save_info=str(i))