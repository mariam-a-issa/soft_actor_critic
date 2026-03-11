from numpy.typing import NDArray
import numpy as np
from torch import Tensor, tensor, device
import torch
from nasimemu import env_utils
from torch_geometric.data import Data

from .env_compat import EnvCompat

from implementation import Agent

class Connector():

    def __init__(self, env : EnvCompat, device : device, env_info : dict):
        self._env = env #Used not to interact with the environment but so that information about the environment can be accessed
        self._device = device

        if 'observation_format' in env_info.keys() and env_info['observation_format'] == 'graph':
            self._graph = True
        else:
            self._graph = False

        if 'flat_obs' in env_info.keys() and env_info['flat_obs']:
            self._flat_obs = True
        else:
            self._flat_obs = False

        self._state = None

        self._env_info = env_info 
        assert not (self._flat_obs and not self._graph)

        self._state_space_dim = None


    def format_state(self, state : NDArray) -> Tensor:
        """Will format the state from the environment to agents standard

        Args:
            state (NDArray): State from environment

        Returns:
            Tensor: Reformatted state
        """

        #Stuff from nasimemu
        # if graph:
        #     return Data(tensor(s[0], dtype=float32), tensor(s[1], dtype=int64)) #0 is node feats and 1 is edge_index. Need to have the data types so that they match up with the rest of the model
        # return  tensor(s[:-1])

        if self._flat_obs:
            return tensor(state, device=self._device)
        else:
            self._state = state

        return self._clean_state(state)
    
    def format_action(self, action : Tensor) -> "EnvAction":
        """Will format the action from the agent to the environments standard

        Args:
            action (int): The index of the action that will take place

        Returns:
            EnvAction: Action formatted for the current environment
        """

        #Stuff from nasimemu
        # if explore_steps <= steps:
        #     action = agent.sample(state) 
        #     return convert_int_action(action.data, env, state, graph), action
        # else:
        #     action = random.randint(0, env.action_space.n-1) #Fix so that it takes into account padded actions depending on size of state
        #     return convert_int_action(action, env, state, graph), tensor(action) 

        if self._flat_obs:
            return int(action)
        else:
            return self._convert_int_action(action)
    
    def get_env_info(self) -> tuple[int, int, EnvCompat]:
        """Will get the state and action space of the environment

        Returns:
            tuple[int, int]: (state_space, action_space)
        """

        if self._flat_obs:
            return self._env.observation_space.shape[0] ,self._env.action_space.n, self._env

        env = self._env.env
        new_env = self._env
        #Got from their config file on how to get sorta of an idea of the size of state and action spaces
        action_space = len(env.action_list)

        if self._state_space_dim is None:
            s = env.reset()
            state_space = s.shape[1] #- MAGIC_CORP_NUM TODO if going to clean put this back in
            self._state_space_dim = state_space
        else:
            state_space = self._state_space_dim
    
        if self._graph and self._env_info['observation_format'] != 'graph_v2':
            self._env_info['observation_format'] = 'graph_v2'
            new_env = EnvCompat.make(**self._env_info)
            new_env.reset()
            state_space += 1 # +1 feature (node/subnet) from NASimEmu Agents and seems to be used only when using graphs
            env = new_env.env

        state_space -= env.env.env.scenario.address_space_bounds[0] + env.env.env.scenario.address_space_bounds[1]
        
        return action_space, state_space, new_env 


    
    def _convert_int_action(self, action : int,) -> tuple[tuple[int, int], int]:
        """Will convert the integer action to a tuple containing device and action information 

        Args:
            action (int): The current action index
            s (Tensor | Data): The current state

        Returns:
            tuple[tuple[int, int], int]: Tuple containg device and action
        """
        s = self._state
        if not self._graph:
            aux_row = np.zeros((1, s.shape[1])) #Needed since possible actions assumes that there is the auxillary data that was cut out earlier
            
            if isinstance(s, Tensor):
                np_s = s.cpu().numpy()
            else:
                np_s = s
                
            np_s = np.concatenate((np_s, aux_row), axis=0)
        else:
            np_s = s.x[:, 1:].cpu().numpy()
            
        return env_utils.get_possible_actions(self._env.env, np_s)[action]
    
    def _clean_state(self, s : NDArray | tuple) -> Tensor | Data:
        """Will clean up the state and return it.
            Many of the NASimEmu agents do not use the additonal information row (data about whether an action was successful)"""
        
        env = self._env.env

        address_size = env.env.env.scenario.address_space_bounds[0] + env.env.env.scenario.address_space_bounds[1]
        if self._graph:
            feats = s[0]
        else:
            feats = s
        mask = np.ones(feats.shape[1], dtype=bool)
        mask[1 : address_size + 1] = False  # exclude columns 1 through n
        if self._graph:
            return Data(tensor(s[0][:, mask], dtype=torch.float32, device=self._device), tensor(s[1], dtype=torch.int64, device=self._device)) #0 is node feats and 1 is edge_index. Need to have the data types so that they match up with the rest of the model
        return tensor(s[:-1, mask], device=self._device)