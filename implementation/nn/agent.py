from pathlib import Path
from copy import deepcopy
import random

from torch import Tensor
import torch

from utils import MemoryBuffer, Transition, Config
from .implementation import Actor, Alpha, QFunction, QFunctionTarget
from ..agents import Agent
from ..model_utils import pad, reshape, generate_batch_index
from .. import sac

class MLPNNAgent(Agent):

    def __init__(self, state_dim : int, action_dim : int, config : Config):
        super().__init__(config.target_update, config.update_frequency, config.learning_steps)
    
        self._input_size = state_dim 
        output_size = action_dim
        
        self._target_q = QFunctionTarget(None, config.tau)
        self._alpha = Alpha(output_size,
                            config.alpha_value,
                            config.alpha_lr, 
                            config.autotune)

        self._policy = Actor(state_dim, 
                    action_dim, 
                    config.hidden_dim,
                    self._target_q,
                    self._alpha, 
                    config.policy_lr,
                    config.grad_clip)
            
        self._q_function = QFunction(state_dim,
                                action_dim,
                                config.hidden_dim,
                                self._policy,
                                self._target_q,
                                self._alpha,
                                config.critic_lr,
                                config.discount,
                                config.grad_clip)
            
        self._target_q.set_actual(self._q_function)

        if config.gpu:
            device = torch.device(f'cuda:{config.gpu_device}')
        else:
            device = torch.device('cpu')
        
        for obj in [self._target_q, self._alpha, self._policy, self._q_function]:
            obj.to(device)
            
        self._memory = MemoryBuffer(config.buffer_size, config.sample_size, random)
        
    def param_update(self) -> dict[str : float]:
        trans = self._memory.sample()
        #TODO Move the learning logic here
        
        q_info = self._q_function.update(trans)
        actor_info = self._policy.update(trans)
        
        info_dict = {
            'QFunc1 Loss' : q_info[0].item(),
            'QFunc2 Loss' : q_info[1].item(),
            'Actor Loss' : actor_info[0].item(),
            'Entropy' : actor_info[1].item(),
            'Alpha Loss' : actor_info[2].item(),
            'Alpha' : actor_info[3].item()
        }
        
        return info_dict
    
    def target_param_update(self):
        self._target_q.update()
    
    def sample(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            action, _, _ = self._policy(state)
            return action
    
    def evaluate(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            action = self._policy.evaluate(state)
            return action
        
    def add_data(self, trans : Transition) -> None:
        self._memory.add_data(trans)
        
    def save(self) -> None:
        pass
    
