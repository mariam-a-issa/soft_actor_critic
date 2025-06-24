import random

import torch
from torch import Tensor, tensor, float32, int64
from torch_geometric.data import Data
import numpy as np
from nasimemu import env_utils
from numpy.typing import NDArray
from nasimemu.nasim.envs.host_vector import HostVector
import gym

from implementation import Agent
from utils import Config


def clean_state(s: NDArray | tuple, graph: bool) -> Tensor | Data:
    """Will clean up the state and return it.
        Many of the NASimEmu agents do not use the additional information row (whether an action was successful)"""
    if graph:
        # 0 is node features and
        # 1 is edge_index.
        # Data type included to match the rest of model
        return Data(tensor(s[0], dtype=float32), tensor(s[1], dtype=int64))
    return tensor(s[:-1])


def get_action(state: Tensor,
               env: gym.Env,
               agent: Agent,
               graph: bool,
               explore_steps: int,
               steps: int) -> tuple[tuple[tuple[int, int], int], Tensor]:
    """Decides action: explores or utilizes policy
        Returns NASimEmu action and integer action as a Tensor"""
    if explore_steps <= steps:
        action = agent.sample(state) 
        return convert_int_action(action.data, env, state, graph), action
    else:
        # Fix: takes into account padded actions depending on size of state
        action = random.randint(0, env.action_space.n-1)
        return convert_int_action(action, env, state, graph), tensor(action) 


def get_train_env_info(env: gym.Env, config: Config) -> tuple[int, int, gym.Env]:
    """ Returns environment's: action space size, state space size

    Args:
        env (gym.Env): Current environment
        config (Config): The config object of the current training run

    Returns:
        tuple[int, int, gym.Env]: action_space_size, state_space_size, remade env
    """
    if config.environment_info['id'] == 'NASimEmu-v0':
        action_space_size = len(env.action_list)
        s = env.reset()
        state_space_size = s.shape[1] #- MAGIC_CORP_NUM TODO if going to clean put this back in
    
    if config.graph:
        config.environment_info['observation_format'] = 'graph_v2'
        env = gym.make(**config.environment_info)
        env.reset()
        state_space_size += 1 # +1 feature (node/subnet) from NASimEmu Agents and seems to be used only when using graphs # TODO: MARIAM -- why this logic?
        action_space_size = len(env.action_list) # TODO: Mariam -- added this in, just copied it over, not sure if it's correct
    return action_space_size, state_space_size, env


def setup_env(config: Config) -> torch.device:
    """Sets up the computer training environment

    Args:
        config (Config): The config object of the current training run

    Returns:
        torch.device: The device object that the models will be running on
    """

    if config.seed is not None:
        torch.manual_seed(config.seed) 
        torch.use_deterministic_algorithms(True, warn_only=True)
        random.seed(config.seed)
        np.random.seed(config.seed)

    if torch.cuda.is_available() and config.gpu:
        device = f'cuda:{config.gpu_device}'
    else:
        device = 'cpu'

    device_obj = torch.device(device)
    torch.set_default_device(device_obj)
    return device_obj


def convert_int_action(action:  int, env, s: Tensor | Data, graph: bool) -> tuple[tuple[int, int], int]:
    """Converts integer from policy --> tuple containing device and action"""
    
    if not graph:
        # Needed since possible actions assumes that there is the auxiliary data that was cut out earlier
        aux_row = np.zeros((1, s.shape[1]))
        
        if isinstance(s, Tensor):
            np_s = s.cpu().numpy()
        else:
            np_s = s
            
        np_s = np.concatenate((np_s, aux_row), axis=0)
    else:
        np_s = s.x[:, 1:].cpu().numpy()
        
    return env_utils.get_possible_actions(env, np_s)[action]
