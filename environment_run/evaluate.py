#import numpy as np
import gymnasium as gym
import torch

from discrete import Agent
from utils import LearningLogger
from .helpers import convert_int_action, clean_state

def evaluate(env : gym.Env, agent : Agent, num_eval, cur_epi, graph) -> None:
    """Will evaluate the current agent on the environment for a given amount of episodes and then log the results"""
    
    epi_reward = 0
    
    for i in range(num_eval):
        
        done = False
        state = clean_state(env.reset(), graph)
        
        while not done:
            action = agent.evaluate(state)
            next_state, reward, done, _ = env.step(convert_int_action(action.data, env, state, graph))
            next_state = clean_state(next_state, graph)
            if reward >0:
                print(f'In eval {i} reward of {reward}')
            epi_reward += reward
            state = next_state
            
    LearningLogger().log_scalars({'Episodic Reward' : epi_reward / num_eval, 'Episode' : cur_epi}, episodes=cur_epi)       
    
            