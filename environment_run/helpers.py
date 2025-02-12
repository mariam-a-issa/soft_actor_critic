from torch import Tensor, tensor, float32, int64
from torch_geometric.data import Data
import numpy as np
from nasimemu import env_utils
from numpy.typing import NDArray
from nasimemu.nasim.envs.host_vector import HostVector


def convert_int_action(action : int, env, s : Tensor | Data, graph : bool) -> tuple[tuple[int, int], int]:
    """Will convert an integer from policy into tuple containing device and action to do at device"""
    
    if not graph:
        aux_row = np.zeros((1, s.shape[1])) #Needed since possible actions assumes that there is the auxillary data that was cut out earlier
        
        if isinstance(s, Tensor):
            np_s = s.cpu().numpy()
        else:
            np_s = s
            
        np_s = np.concatenate((np_s, aux_row), axis=0)
    else:
        np_s = s.x[:, 1:].cpu().numpy()
        
    return env_utils.get_possible_actions(env, np_s)[action]

def clean_state(s : NDArray | tuple, graph : bool) -> Tensor | Data:
    """Will clean up the state and return it.
        Many of the NASimEmu agents do not use the additonal information row (data about whether an action was successful)"""
    if graph:
        return Data(tensor(s[0], dtype=float32), tensor(s[1], dtype=int64)) #0 is node feats and 1 is edge_index. Need to have the data types so that they match up with the rest of the model
    return tensor(s[:-1])