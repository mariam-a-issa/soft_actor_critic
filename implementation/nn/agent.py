from pathlib import Path
from copy import deepcopy
import random

from torch import Tensor
import torch
from torch.nn import utils

from utils import MemoryBuffer, Transition, Config
from .implementation import Actor, QFunction, QFunctionTarget
from ..agents import Agent
from .. import sac

class MLPNNAgent(Agent):

    def __init__(self, state_dim : int, action_dim : int, config : Config):
        super().__init__(config.target_update, config.update_frequency, config.learning_steps)
    
        self._input_size = state_dim 
        
        self._q_func_target = QFunctionTarget(None, config.tau)

        self._policy = Actor(state_dim, 
                    action_dim, 
                    config.hidden_dim)
            
        self._q_func = QFunction(state_dim,
                                action_dim,
                                config.hidden_dim)
        self._alpha = sac.Alpha(start=config.target_entropy_start, 
                    end=config.target_entropy_end, 
                    midpoint=config.target_entropy_midpoint, 
                    slope=config.target_entropy_slope, 
                    max_steps=config.max_steps, 
                    autotune=config.autotune, 
                    alpha_value=config.alpha_value)
            
        self._q_func_target.set_actual(self._q_func)

        if config.gpu:
            device = torch.device(f'cuda:{config.gpu_device}')
        else:
            device = torch.device('cpu')
        
        for obj in [self._q_func_target, self._alpha, self._policy, self._q_func]:
            obj.to(device)

        self._optim_critic = torch.optim.Adam([*self._q_func.parameters()], lr=config.critic_lr)
        self._optim_policy = torch.optim.Adam([*self._policy.parameters()], lr=config.policy_lr)
        self._optim_alpha = torch.optim.Adam([self._alpha._log_alpha], lr = config.alpha_lr)
            
        self._memory = MemoryBuffer(config.buffer_size, config.sample_size, random)

        self._action_dim = torch.tensor(action_dim)
        self._config = config
        
    def param_update(self) -> dict[str : float]:
        trans = self._memory.sample()
        
        cur_q1 : Tensor
        cur_q2 : Tensor
        cur_q1, cur_q2 = self._q_func.q_values(trans.state)
        _, cur_prob, cur_log_prob = self._policy(trans.state)
        
        cur_log_prob = cur_log_prob / torch.log(self._action_dim) #Normilize by the maximum possible entropy
        
        with torch.no_grad():
            
            q_target = self._q_func_target(trans.state)
            next_q_target = self._q_func_target(trans.next_state)
            _, next_prob, next_log_prob = self._policy(trans.next_state)
            
            next_log_prob = next_log_prob / torch.log(self._action_dim) #Normilize by the maximum possible entropy
            batch_size, cur_action_size = cur_prob.shape
            
            ent = -torch.bmm(cur_prob.view(batch_size, 1, cur_action_size),
                            cur_log_prob.view(batch_size, cur_action_size, 1)).mean()
        
        
        policy_loss = sac.policy_loss(q_target, cur_prob, cur_log_prob, self._alpha()).mean().squeeze()
        q1_dif, q2_dif = sac.q_func_loss(cur_q1, 
                                         cur_q2,
                                         next_q_target,
                                         trans.action.view(-1,1),
                                         next_prob,
                                         next_log_prob,
                                         trans.reward,
                                         self._alpha(),
                                         self._config.discount,
                                         trans.done)
        alpha_loss = sac.alpha_loss(cur_prob,
                                    cur_log_prob,
                                    self._alpha(),
                                    self._alpha.sigmoid_target_entropy())
        
        q1_loss = sac.mse(q1_dif)
        q2_loss = sac.mse(q2_dif)
        
        self._optim_policy.zero_grad()
        policy_loss.backward()
        
        critic_loss = q1_loss + q2_loss
        
        self._optim_critic.zero_grad()
        critic_loss.backward()
        
        grad_policy = self.calc_grad_norm([*self._policy.parameters()])
        grad_q_func = self.calc_grad_norm([*self._q_func.parameters()])

        self._optim_alpha.zero_grad()
        alpha_loss.backward()
        
        if self._config.grad_clip:
            utils.clip_grad_norm_([*self._q_func.parameters()], self._config.grad_clip)
        
        self._optim_policy.step()
        self._optim_critic.step()
        self._optim_alpha.step()
        

        return {
            'QFunc1 Loss' : q1_loss.item(),
            'QFunc2 Loss' : q2_loss.item(),
            'Actor Loss' : policy_loss.item(),
            'Alpha Loss' : alpha_loss.item(),
            'Entropy' : ent.item(),
            'Alpha Value' : self._alpha().item(),
            'Grad of Policy' : grad_policy,
            'Unclipped Grad of Q Func' : grad_q_func
        }
    
    def target_param_update(self):
        self._q_func_target.update()
    
    def sample(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            action = self._policy.sample(state)
            return action
    
    def evaluate(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            action = self._policy.evaluate(state)
            return action
        
    def add_data(self, trans : Transition) -> None:
        self._memory.add_data(trans)
        
    def save(self) -> None:
        pass
    
