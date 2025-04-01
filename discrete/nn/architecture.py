from pathlib import Path
import os

import torch
from torch import Tensor, nn

class BaseNN(nn.Module):
    """Base class for constructing NNs"""

    def __init__(self, input_size : int,  output_size : int, hidden_size : list[int], id : int = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._hidden_size = hidden_size

        layer_list = [nn.Linear(input_size, hidden_size[0]), nn.ReLU()]
        
        for i, size in enumerate(hidden_size[:-1]):
            layer_list.extend([
                nn.Linear(size, hidden_size[i+1]),
                nn.ReLU()
            ])
        
        layer_list.append(nn.Linear(hidden_size[-1], output_size))
        
        self.layers = nn.Sequential(*layer_list)
        
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
