from torch import Tensor
import torch

from utils import MemoryBuffer, Transition, Config, PrioritizedMemoryBuffer
from .implementation import Actor, QFunction, QFunctionTarget
from .encoders import EXPEncoder, RBFEncoder
from ..agents import Agent
from .. import sac

class HDCAgent(Agent):

    def __init__(self, state_dim : int, action_dim : int, config : Config):
        super().__init__(config.target_update, config.update_frequency, config.learning_steps)

        if config.gpu:
            device = torch.device(f'cuda:{config.gpu_device}')
        else:
            device = torch.device('cpu')
    
        self._input_size = state_dim 

        self._policy_enc = RBFEncoder(state_dim, config.hypervec_dim)
        self._q_func_enc = EXPEncoder(state_dim, config.hypervec_dim)

        self._policy = Actor(config.hypervec_dim, 
                            action_dim)
            
        self._q_func = QFunction(config.hypervec_dim,
                                action_dim)
        
        self._q_func_target = QFunctionTarget(config.tau, self._q_func)
        
        self._alpha = sac.Alpha(start=config.target_entropy_start, 
                    end=config.target_entropy_end, 
                    midpoint=config.target_entropy_midpoint, 
                    slope=config.target_entropy_slope, 
                    max_steps=config.max_steps, 
                    autotune=config.autotune, 
                    alpha_value=config.alpha_value,
                    device=device)

        
        for obj in [self._q_func_target, self._policy, self._q_func, self._q_func_enc, self._policy_enc]:
            obj.to(device)

        self._optim_policy = torch.optim.Adam([*self._policy.parameters()], lr=config.policy_lr)
        self._optim_alpha = torch.optim.Adam([self._alpha._log_alpha], lr=config.alpha_lr)
        
        self._prioritized = config.prioritized_alpha is not None and config.prioritized_beta is not None

        if self._prioritized:
            self._memory = PrioritizedMemoryBuffer(buffer_length=config.buffer_size, 
                                                   sample_size=config.sample_size, 
                                                   alpha=config.prioritized_alpha,
                                                   beta=config.prioritized_beta,
                                                   device=device)
        else:
            self._memory = MemoryBuffer(buffer_length=config.buffer_size, 
                                        sample_size=config.sample_size)

        self._action_dim = torch.tensor(action_dim, device=device)
        self._config = config
        
    def param_update(self) -> dict[str : float]:
        trans = self._memory.sample()
        
        with torch.no_grad():
            cur_p_emb = self._policy_enc(trans.state)

        _, cur_prob, cur_log_prob = self._policy(cur_p_emb)

        with torch.no_grad():
            cur_q_emb = self._q_func_enc(trans.state)
            next_q_emb = self._q_func_enc(trans.next_state)
    
            next_p_emb = self._policy_enc(trans.next_state)

            q_target = self._q_func_target(cur_q_emb)
            next_q_target = self._q_func_target(next_q_emb)
            _, next_prob, next_log_prob = self._policy(next_p_emb)
            
            next_log_prob = next_log_prob / torch.log(self._action_dim) #Normilize by the maximum possible entropy
            batch_size, cur_action_size = cur_prob.shape
            
            ent = -torch.bmm(cur_prob.view(batch_size, 1, cur_action_size),
                            cur_log_prob.view(batch_size, cur_action_size, 1)).mean()
            
            
            cur_q1, cur_q2 = self._q_func(cur_q_emb)

        _, cur_prob, cur_log_prob = self._policy(cur_p_emb)
        cur_log_prob = cur_log_prob / torch.log(self._action_dim) #Normilize by the maximum possible entropy
        
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
        
        if self._config.autotune:
            alpha_loss = sac.alpha_loss(cur_prob,
                                        cur_log_prob,
                                        self._alpha(),
                                        self._alpha.sigmoid_target_entropy())
            self._optim_alpha.zero_grad()
            alpha_loss.backward()
            self._optim_alpha.step()
            alpha_dict = {'Alpha Value' : self._alpha().item(), 'Alpha Loss' : alpha_loss.item()}
        else:
            alpha_dict = {}
        
        
        self._optim_policy.zero_grad()
        policy_loss.backward()
        
        with torch.no_grad():
            matrix_l1 = q1_dif * cur_q_emb * self._config.critic_lr
            matrix_l2 = q2_dif * cur_q_emb * self._config.critic_lr
            #Index add will add the vector found at index i of matrix_l1 to index a_i of the model (returned by parameters()),
            #where a_i is the value of trans.action at index i
            #trans.action is a b x 1 column vector but needs to be row vector so squeeze
            self._q_func._q1.parameters().index_add_(0, trans.action.squeeze(), matrix_l1)
            self._q_func._q2.parameters().index_add_(0, trans.action.squeeze(), matrix_l2)
        
        
        grad_policy = self.calc_grad_norm([*self._policy.parameters()])

        self._optim_policy.step()

        if self._prioritized:
            self._memory.update_priority(trans, (q1_dif + q2_dif).detach().abs())
        
        return {
            'QFunc1 Loss' : sac.mse(q1_dif).item(),
            'QFunc2 Loss' : sac.mse(q2_dif).item(),
            'Actor Loss' : policy_loss.item(),
            'Entropy' : ent.item(),
            'Grad of Policy' : grad_policy,
        } | alpha_dict
    
    def target_param_update(self):
        self._q_func_target.update()
    
    def sample(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            emb = self._policy_enc(state)
            action, _, _ = self._policy(emb)
            return action
    
    def evaluate(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            emb = self._policy_enc(state)
            action = self._policy.evaluate(emb)
            return action
        
    def add_data(self, trans : Transition) -> None:
        self._memory.add_data(trans)
        
    def save(self) -> None:
        pass
    
