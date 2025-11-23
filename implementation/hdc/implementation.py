from copy import deepcopy
from pathlib import Path
import os
import math

from torch import nn, Tensor, optim
import torch
from torch.distributions import Categorical
import torch.nn.functional as F 

from .encoders import RBFEncoder, EXPEncoder
from utils.data_collection import Transition
from utils import MAX_ROWS, NEG_INF 

class QModel:

    def __init__(self, hvec_dim : int, action_dim : int) -> None:
        """Will create a model that is a matrix that contains a hypervector for each action"""
        upper_bound = 1 / math.sqrt(hvec_dim)
        lower_bound = -upper_bound
        
        #Using the same initilzation as the torch.nn.Linear 
        #https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L106-L108

        self._model = (upper_bound - lower_bound) * torch.rand(action_dim, hvec_dim, dtype=torch.cfloat) + lower_bound
        self._model.requires_grad_(False)
        self._hdvec_dim = hvec_dim
        self._action_dim = action_dim

    def __call__(self, state : Tensor) -> Tensor:
        """Parameter is a batch of encoded states and will 
        return the batch of vectors where each element is the actions q value
        
        b x hd -> b x a

        """
        
        # Need to broadcast model to batched state so state needs to be unsqueezed
        with torch.no_grad():
            return torch.real((torch.conj(self._model) @ state.unsqueeze(dim = 2)).squeeze() / self._hdvec_dim).view(state.shape[0], self._action_dim)
    
    def parameters(self) -> Tensor:
        return self._model
    
    def to(self, dev : torch.device) -> None:
        self._model = self._model.to(dev)
    

class QFunction:

    def __init__(self, hvec_dim : int, action_dim : int) -> None:
        """Will create a Q function that has two q models"""
        
        self._q1 = QModel(hvec_dim, action_dim)
        self._q2 = QModel(hvec_dim, action_dim)


    def __call__(self, state) -> tuple[Tensor, Tensor]:
        """State should be an encoded h_vect"""
        return self._q1(state), self._q2(state)

    def to(self, device : torch.device) -> None:
        """Moves q function to device"""
        self._q1.to(device)
        self._q2.to(device)

class QFunctionTarget:
    
    def __init__(self,
                 tau : int,
                 q_function : QFunction) -> None:

        self._actual = q_function

        if q_function is not None:
            self._q1 = deepcopy(q_function._q1)
            self._q2 = deepcopy(q_function._q2)

        self._tau = tau

    def set_actual(self, q_function : QFunction) -> None:
        """Will actually set the q_function if it was not set in init"""
        self._actual = q_function

        self._q1 = deepcopy(q_function._q1)
        self._q2 = deepcopy(q_function._q2)
    
    def __call__(self, state) -> Tensor:
        return torch.min(self._q1(state), self._q2(state))

    def update(self) -> None:
        """Will do polyak averaging to each model in the target"""
        for param, target_param in zip(self._actual._q1.parameters(), self._q1.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._actual._q2.parameters(), self._q2.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
    
    def to(self, dev : torch.device) -> None:
        self._q1.to(dev)
        self._q2.to(dev)

class Actor(nn.Module):

    def __init__(self, hvec_dim : int, action_dim : int) -> None:
        super().__init__()
        
        self._action_s = action_dim #Amount of actions per device    
        self._logits = nn.Linear(hvec_dim, action_dim, bias=False)
        self._logits.weight.data = torch.zeros((action_dim, hvec_dim))
    

    def forward(self, state : Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Will give the action, log_prob, action_probs of action
           If padding was done, then in the batch there would be states with various lengths which will need to be taken account of when doing"""

        logits : Tensor = self._logits(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_probs = dist.probs
        log_prob = F.log_softmax(logits, dim=-1)
        return action, log_prob, action_probs
    
    def evaluate(self, state : Tensor) -> Tensor:
        """Will return the best action for evaulation"""
        
        return torch.argmax(self._logits(state))
        
    def save(self, file_name ='best_weights.pt') -> None:
        """Will save the model in the folder 'model' in the dir that the script was run in."""

        folder_name = type(self).__name__

        model_folder_path = Path('./model/' + folder_name)
        file_dir = Path(os.path.join(model_folder_path, file_name))

        if not os.path.exists(file_dir.parent):
            os.makedirs(file_dir.parent)

        torch.save(self.state_dict(), file_dir)



