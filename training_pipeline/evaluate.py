import gymnasium as gym

from implementation import Agent
from utils import LearningLogger
from .env import Connector, EnvCompat

def evaluate(env : EnvCompat, agent : Agent, num_eval : int, cur_epi : int, graph : bool) -> None:
    """Will evaluate the current agent on the environment for a given amount of episodes and then log the results"""
    
    epi_reward = 0

    ctr = Connector(env)
    
    for i in range(num_eval):
        
        done = False
        state = ctr.format_state(env.reset()[0])

        while not done:
            action = agent.evaluate(state)
            action_nas = ctr.format_action(action)
            next_state, reward, done, step_limit_reached, _ = env.step(action_nas)
            done = done or step_limit_reached
            next_state = ctr.format_state(next_state)
            if reward >0:
                print(f'In eval {i} reward of {reward}')
            epi_reward += reward
            state = next_state
            
    LearningLogger().log_scalars({'Episodic Reward' : epi_reward / num_eval, 'Episode' : cur_epi}, episodes=cur_epi)       
    
            