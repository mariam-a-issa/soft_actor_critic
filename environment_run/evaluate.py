#import numpy as np
import gymnasium as gym
import torch

from discrete import Agent
from utils import LearningLogger

def evaluate(env : gym.Env, agent : Agent, num_eval, cur_epi) -> None:
    """Will evaluate the current agent on the environment for a given amount of episodes and then log the results"""
    
    for _ in range(num_eval):
        
        done = False
        state = env.reset()[0]
        epi_reward = 0
        
        while not done:
            action = agent.evaluate(torch.from_numpy(state).to(torch.get_default_device()))
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated
            epi_reward += reward
            state = next_state
            
    LearningLogger().log_scalars({'Episodic Reward' : epi_reward / num_eval, 'Episode' : cur_epi}, episodes=cur_epi)       
    
            