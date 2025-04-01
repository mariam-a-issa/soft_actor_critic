from pathlib import Path
from copy import deepcopy

from torch import Tensor
import torch

from utils import DynamicMemoryBuffer, Transition, Config, MAX_ROWS
from .implementation import Actor, Alpha, QFunction, QFunctionTarget
from ..agents import Agent
from ..model_utils import pad, reshape, generate_batch_index
from .. import sac

class MLPNNAgent(Agent):

    def __init__(self, node_dim : int, action_dim : int, config : Config):
        super().__init__(config.target_update, config.update_frequency, config.learning_steps)
    
        self._input_size = node_dim * MAX_ROWS
        output_size = action_dim * MAX_ROWS
        
        self._target_q = QFunctionTarget(None, config.tau)
        self._alpha = Alpha(output_size,
                            config.alpha_value,
                            config.alpha_lr, 
                            config.autotune)

        self._policy = Actor(node_dim, 
                    action_dim, 
                    config.hidden_dim,
                    self._target_q,
                    self._alpha, 
                    config.policy_lr,
                    config.grad_clip)
            
        self._q_function = QFunction(node_dim,
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
            
        self._memory = DynamicMemoryBuffer(config.buffer_size, config.sample_size)
        
    def param_update(self) -> dict[str : float]:
        trans = self._memory.sample()
        #TODO Make sure that the reshaped is going to be the same as the input size with some type of modificiation to pad function
        
        reshape_state = reshape(trans.state, generate_batch_index(trans.state_index), filler_val=0)
        reshape_next_state = reshape(trans.next_state, generate_batch_index(trans.next_state_index), filler_val=0)
        
        batch_size, _ = reshape_state.shape
        
        pad_state = torch.zeros(batch_size, self._input_size)
        pad_next_state = torch.zeros(batch_size, self._input_size)
        
        pad_state[:, :reshape_state.shape[1]] = reshape_state
        pad_next_state[:, :reshape_next_state.shape[1]] = reshape_next_state
        
        trans = Transition(state = pad_state,
                next_state = pad_next_state,
                action=trans.action,
                reward=trans.reward,
                done=trans.done,
                num_devices=trans.num_devices,
                num_devices_n=trans.num_devices_n)
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
            padded_state = pad(state, self._input_size)
            #Even though this looks like it would work with dynamic False I do not think it would. Should just assume it is dynamic for this branch tbh
            action, _, _ = self._policy(padded_state, num_devices = torch.tensor(state.shape[0]).unsqueeze(dim=0), batch_size = 1)
            return action
    
    def evaluate(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            padded_state = pad(state, self._input_size)
            #Even though this looks like it would work with dynamic False I do not think it would. Should just assume it is dynamic for this branch tbh
            action = self._policy.evaluate(padded_state, num_devices = torch.tensor(state.shape[0]).unsqueeze(dim=0))
            return action
        
    def add_data(self, trans : Transition) -> None:
        self._memory.add_data(trans)
        
    def save(self) -> None:
        pass
    