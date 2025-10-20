from abc import ABC, abstractmethod
from typing import Iterable

from torch import Tensor
from torch_geometric.data import Data

from utils import Transition, LearningLogger
    
class Agent(ABC):
    
    def __init__(self,
                 target_update : int,
                 update_frequency : int,
                 learning_steps : int):
        
        self._learning_steps = learning_steps
        self._update_frequency = update_frequency
        self._target_update = target_update
    
    @abstractmethod
    def sample(self, state : Tensor | Data) -> Tensor:
        """Sample an action given a state

        Args:
            state (Tensor | Data): The state representation that will be sampled

        Returns:
            Tensor: The integer index of the action to be sampled
        """
        pass
    
    @abstractmethod
    def evaluate(self, state : Tensor | Data) -> Tensor:
        """Return the most likely action given a state

        Args:
            state (Tensor | Data): The state representation that will be sampled

        Returns:
            Tensor: The integer index of the action to be sampled
        """
        pass

    @abstractmethod
    def save(self, directory : str) -> None:
        """Will save the weights of the models into the given directory 

        Args:
            extension (str): The name of the directory
        """
        pass

    @abstractmethod  
    def add_data(self, trans : Transition) -> None:
        """Adds data from a given transition into memory

        Args:
            trans (Transition): The step transition to add to memory
        """
        pass
    
    @abstractmethod
    def param_update(self) -> dict[str : float]:
        """Will do a parameter update and return a dictionary of values that should be logged

        Returns:
            dict[str : float]: A dictionary mapping the name of the value and the value itself to be logged
        """
        pass

    @abstractmethod
    def target_param_update(self) -> None:
        """Will do the correct update for the target networks
        """
        pass
    
    def update(self, steps : int) -> None:
        """Will update the parameters of the models according to the current step in training

        Args:
            steps (int): The current step in training
        """
        if steps % self._update_frequency == 0:
            
            log_dicts = []
            
            for _ in range(self._learning_steps):
                log_dicts.append(self.param_update())
                
            self._log_data(log_dicts, steps)
            
        if steps % self._target_update == 0:
            self.target_param_update()

    def calc_grad_norm(self, parameters : Iterable[Tensor]) -> float:
        """Will calculate the norm of the gradient across the parameters

        Args:
            parameters (Iterable[Tensor]): Parameters of a given network

        Returns:
            float: The norm of the parameters
        """
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    
    def polyak_average(self, actual_params : Iterable[Tensor], target_params : Iterable[Tensor], tau) -> None:
        """Will do a polyak average to update to update the target parameters given the actual parameters

        Args:
            actual_params (Iterable[Tensor]): Used to update
            target_params (Iterable[Tensor]): Will be updated
            tau (_type_): How much to update
        """
        for param, target_param in zip(actual_params, target_params):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _log_data(self, log_dicts: list[dict[str : float]], steps : int) -> None:
        new_log_dict = dict()
        number_dicts = len(log_dicts)

        for log_dict in log_dicts:
            for key, value in log_dict.items():
                new_log_dict[key] = value / number_dicts

        logger = LearningLogger()
        logger.log_scalars(new_log_dict, steps=steps)
    