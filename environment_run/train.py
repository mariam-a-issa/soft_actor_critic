import random
from copy import deepcopy

import torch
import gym
import numpy as np

from discrete import MILHDCAgent, Agent
from utils import Transition, LearningLogger, Config
from .evaluate import evaluate
from .helpers import clean_state, get_action

LOG_DIR = 'runs'
MAGIC_CORP_NUM = 20 #Magic number involved with indexing in the corp scenario

def train(
        run_name : str = '',  #Multiple runs inside of a job (Usually for different seeds)
        base_dir : str = LOG_DIR, #Root of all experiments
        group_name : str = '', #Groups of various experiments
        job_name : str = '',   #Individual jobs in the experiment
        config : Config = None) -> None:
    """Will be the main training loop"""
    
    h_params_dict = config.to_flat_dict()

    logger = LearningLogger(base_dir, group_name, job_name, run_name, h_params_dict, tensorboard=config.tensorboard, wandb=config.wandb, save_csv=config.save_csv)
    
    #"LunarLander-v2"
    #"CartPole-v1"
    #"MountainCar-v0"
    env : gym.Env
    env = gym.make(**config.environment_info)
    env.reset()
    
    if config.environment_info['id'] == 'NASimEmu-v0':
        #Got from their config file on how to get sorta of an idea of the size of state and action spaces
        action_space = len(env.action_list)
        s = env.reset()
        state_space = s.shape[1] #- MAGIC_CORP_NUM TODO if going to clean put this back in
        dynamic = True
    else:
        action_space = env.action_space.n
        state_space = env.observation_space.shape[0]
        dynamic = False
    
    if config.graph:
        config.environment_info['observation_format'] = 'graph_v2'
        env = gym.make(**config.environment_info)
        env.reset()
        state_space += 1 # +1 feature (node/subnet) from NASimEmu Agents and seems to be used only when using graphs
    
    if config.seed is not None:
        torch.manual_seed(config.seed) 
        torch.use_deterministic_algorithms(True, warn_only=True)
        random.seed(config.seed)
        np.random.seed(config.seed)

    if torch.cuda.is_available() and config.gpu:
        device = f'cuda:{torch.cuda.current_device()}'
    else:
       device = 'cpu'

    device_obj = torch.device(device)

    torch.set_default_device(device_obj)

    agent = MILHDCAgent(node_dim=state_space,
                        action_dim=action_space,
                        config=config)

    steps = 0
    num_epi = 0
    epi_reward = 0
    
    state = clean_state(env.reset(), config.graph)
    try:
        while config.max_steps > steps:
            action_nas, action = get_action(state=state, env=env, agent=agent, graph=config.graph, explore_steps=config.explore_steps, steps=steps)
            next_state, reward, done, _ = env.step(action_nas)
            real_state = next_state
            next_state = clean_state(next_state, config.graph)
            trans = Transition( #states will be tensors, actions will be tensor integers, the reward will be a float, and terminated will be a bool
                state=state,
                action=action,
                next_state=next_state,
                reward=torch.tensor([reward], device=device_obj, dtype=torch.float32),
                done=torch.tensor([False], device=device_obj, dtype=torch.float32) #Currently the agent never actually comes to a point where it makes a move that terminates. Therefor done should not be incorporated 
            )
            
            epi_reward += reward

            agent.add_data(trans)
            
            if config.explore_steps <= steps:
                agent.update(steps)

            steps += 1

            if done:
                
                LearningLogger().log_scalars({'Training reward' : epi_reward}, episodes=num_epi)
                
                epi_reward = 0
                if config.explore_steps <= steps:
                    num_epi += 1
                    if num_epi % config.eval_frequency == 0:
                        evaluate(deepcopy(env), agent, config.num_evals, num_epi, config.graph) #Need to deepcopy so that we keep the environment the same when training or else the state the environment will be in will be different from the state that is in next_state

            state = next_state
    except Exception as e:
        print(e)
        raise e
    
    finally:
        agent.save(job_name)
        env.close()
        logger.close()


if __name__ == '__main__':
    for i in range(3):
        train(i, hdc_agent=False)
