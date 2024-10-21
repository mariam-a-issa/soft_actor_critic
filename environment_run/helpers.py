from torch import Tensor
import numpy as np
from nasimemu import env_utils
from numpy.typing import NDArray


def convert_int_action(action : int, env, s : Tensor | NDArray) -> tuple[tuple[int, int], int]:
    """Will convert an integer from policy into tuple containing device and action to do at device"""
    aux_row = np.zeros((1, s.shape[1])) #Needed since possible actions assumes that there is the auxillary data that was cut out earlier
    
    if isinstance(s, Tensor):
        np_s = s.cpu().numpy()
    else:
        np_s = s
        
    np_s = np.concatenate((np_s, aux_row), axis=0)
    return env_utils.get_possible_actions(env, np_s)[action]

def clean_state(s : NDArray) -> NDArray:
    """Will clean up the state and return it.
        Many of the NASimEmu agents do not use the additonal information row (data about whether an action was successful)"""
    return s[:-1, 20:] #Removes last layer and then removes ids that takes the first twenty in 