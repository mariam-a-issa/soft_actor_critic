import random
from copy import deepcopy

import torch
from torch import Tensor
import gym
import numpy as np

from implementation import create_agent, Agent
from utils import Transition, LearningLogger, Config
from .evaluate import evaluate
from .env import EnvCompat, Connector, setup_env

LOG_DIR = 'runs'

def train(base_dir : str = LOG_DIR, #Root of all experiments
          experiment_name : str = '', #Groups of various experiments
          hp_info : str = '',   #Individual jobs in the experiment
          config : Config = None) -> None:
    """Will be the main training loop"""

    logger = LearningLogger(base_dir, experiment_name, hp_info, config)
    
    env : gym.Env
    env = EnvCompat.make(**config.environment_info)
    env.reset()

    ctr = Connector(env)
    
    state_space, action_space = ctr.get_env_info()
    device = setup_env(config)

    agent = create_agent(node_dim=state_space,
                        action_dim=action_space,
                        config=config)

    steps = 0
    num_epi = 0
    epi_reward = 0
    
    state = ctr.format_state(env.reset()[0])
    try:
        while config.max_steps > steps:
            action_nas, action = _get_action(state=state, agent=agent, explore_steps=config.explore_steps, steps=steps, ctr=ctr)
            next_state, reward, done, _, _= env.step(action_nas)
            reward = _clamp(reward, -10, 10) #Quick fix
            next_state = ctr.format_state(next_state)
            trans = Transition( #states will be tensors, actions will be tensor integers, the reward will be a float, and terminated will be a bool
                state=state,
                action=action,
                next_state=next_state,
                reward=torch.tensor([reward], device=device, dtype=torch.float32),
                done=torch.tensor([done], device=device, dtype=torch.float32) #Currently the agent never actually comes to a point where it makes a move that terminates. Therefor done should not be incorporated 
            )
            
            epi_reward += reward
            print(reward)
            agent.add_data(trans)
            
            if config.explore_steps <= steps:
                agent.update(steps)

            steps += 1

            if done:
                
                LearningLogger().log_scalars({'Training reward' : epi_reward}, episodes=num_epi)

                next_state = ctr.format_state(env.reset()[0])

                epi_reward = 0
                if config.explore_steps <= steps:
                    num_epi += 1
                    if num_epi % config.eval_frequency == 0:
                        evaluate(deepcopy(env), agent, config.num_evals, num_epi, config.graph) #Need to deepcopy so that we keep the environment the same when training or else the state the environment will be in will be different from the state that is in next_state

            state = next_state
    finally:
        env.close()
        logger.close()

def _clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo must be <= hi")
    return max(lo, min(x, hi))

def _get_action(explore_steps : int, steps : int, agent : Agent, state : Tensor, ctr : Connector):
    if explore_steps <= steps:
        action = agent.sample(state) 
        return ctr.format_action(action), action
    else:
        _, action_space = ctr.get_env_info()
        action = random.randint(0, action_space-1) 
        return ctr.format_action(action), action
