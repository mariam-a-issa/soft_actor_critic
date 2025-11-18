from numpy.typing import NDArray
import numpy as np
from torch import Tensor, tensor, device
from nasimemu import env_utils
from torch_geometric.data import Data

from .env_compat import EnvCompat

from implementation import Agent

class Connector():

    def __init__(self, env : EnvCompat, device : device):
        self._env = env #Used not to interact with the environment but so that information about the environment can be accessed
        self._graph = False
        self._device = device
        pass


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

        return tensor(state, device=self._device)
    
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

        return int(action)
    
    def get_env_info(self) -> tuple[int, int]:
        """Will get the state and action space of the environment

        Returns:
            tuple[int, int]: (state_space, action_space)
        """

        #Stuff from nasimemu
        # if config.environment_info['id'] == 'NASimEmu-v0':
        #     #Got from their config file on how to get sorta of an idea of the size of state and action spaces
        #     action_space = len(env.action_list)
        #     s = env.reset()
        #     state_space = s.shape[1] #- MAGIC_CORP_NUM TODO if going to clean put this back in
    
        # if config.graph:
        #     config.environment_info['observation_format'] = 'graph_v2'
        #     env = gym.make(**config.environment_info)
        #     env.reset()
        #     state_space += 1 # +1 feature (node/subnet) from NASimEmu Agents and seems to be used only when using graphs


        return self._env.observation_space.shape[0] ,self._env.action_space.n
    
    def _convert_int_action(self, action : int, s : Tensor | Data) -> tuple[tuple[int, int], int]:
        """Will convert the integer action to a tuple containing device and action information 

        Args:
            action (int): The current action index
            s (Tensor | Data): The current state

        Returns:
            tuple[tuple[int, int], int]: Tuple containg device and action
        """
        
        if not self._graph:
            aux_row = np.zeros((1, s.shape[1])) #Needed since possible actions assumes that there is the auxillary data that was cut out earlier
            
            if isinstance(s, Tensor):
                np_s = s.cpu().numpy()
            else:
                np_s = s
                
            np_s = np.concatenate((np_s, aux_row), axis=0)
        else:
            np_s = s.x[:, 1:].cpu().numpy()
            
        return env_utils.get_possible_actions(self._env, np_s)[action]