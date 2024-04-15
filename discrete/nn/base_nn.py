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

    def forward(self, x : Tensor) -> Tensor:
        """Using batchs x should be N x D where N is the number of batches"""
        return self.layers(x)
    
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
