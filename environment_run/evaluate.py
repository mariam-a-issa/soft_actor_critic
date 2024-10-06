#import numpy as np
import gymnasium as gym
import torch

from discrete import Agent
from utils import LearningLogger
from .helpers import convert_int_action, clean_state

def evaluate(env : gym.Env, agent : Agent, num_eval, cur_epi) -> None:
    """Will evaluate the current agent on the environment for a given amount of episodes and then log the results"""
    
    epi_reward = 0
    
    for _ in range(num_eval):
        
        done = False
        state = env.reset()
        
        while not done:
            action = agent.evaluate(torch.tensor(clean_state(state)))
            next_state, reward, done, _ = env.step(convert_int_action(action.data, env, state))
            epi_reward += reward
            state = next_state
            
    LearningLogger().log_scalars({'Episodic Reward' : epi_reward / num_eval, 'Episode' : cur_epi}, episodes=cur_epi)       
    
            