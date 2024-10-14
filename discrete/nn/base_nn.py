from pathlib import Path
import os

import torch
from torch import Tensor, nn

class BaseNN(nn.Module):
    """Base class for constructing NNs"""

    def __init__(self, input_size : int,  output_size : int, hidden_size : int, id : int = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._hidden_size = hidden_size

        self.layers = nn.Sequential(
                                nn.Linear(self.input_size, self._hidden_size),
                                nn.ReLU(),
                                nn.Linear(self._hidden_size, self._hidden_size),
                                nn.ReLU(),
                                nn.Linear(self._hidden_size, self.output_size))
        self._id = None

    def forward(self, state : Tensor, num_devices : Tensor = None, batch_size : int = None) -> Tensor:
        """Using batchs x should be N x D where N is the number of batches"""
        return self.layers(state)
    
    def save(self, file_name ='best_weights.pt') -> None:
        """Will save the model in the folder 'model' in the dir that the script was run in."""

        folder_name = type(self).__name__ + self._extra_info

        model_folder_path = Path('./model/' + folder_name)
        file_dir = Path(os.path.join(model_folder_path, file_name))

        if not os.path.exists(file_dir.parent):
            os.makedirs(file_dir.parent)

        torch.save(self.state_dict(), file_dir)

    @property
    def _extra_info(self):
        """Can be overridden to give any extra information about the NN"""
        if self._id is None:
            return ''
        return self._id


def pad(state : Tensor | list[Tensor], input_size : int) -> Tensor:
    """Will pad the state tensor correctly. Note that here we consider either a single matrix or a batch list of matrices. 
       Either return a single vector representing the state or a matrix representing the batch of states"""
    if isinstance(state, Tensor):
        padded_state = torch.zeros(input_size)
        state = state.flatten()
        padded_state[:len(state)] = state
        return padded_state
    elif isinstance(state, list):
        state = [s.flatten() for s in state] #Slow but need to do it here. Maybe move to data collection but then becomes tricky
        padded_state = torch.zeros(input_size) #Need to pad first one to desired length
        padded_state[:len(state[0])] = state[0]
        state[0] = padded_state
        return torch.nn.utils.rnn.pad_sequence(state, batch_first=True, padding_value=0)
    else:
        raise TypeError("Incorrect state representation for padding")