import gym

from nn_sac import train

for i in range(3):
    env = gym.make('Hopper-v4', render_mode = 'human')
    train(env, 11, 3, reward_scale=5.0, max_steps=10**6)