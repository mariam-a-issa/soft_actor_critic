from pathlib import Path

from torch import Tensor
import torch
from torch import optim

from utils import DynamicMemoryBuffer, Transition, Config
from .implementation import Encoder, Actor, QFunction, QFunctionTarget
from ..mil_nn.implementation import Alpha
from ..agents import Agent
from .. import sac


class MILHDCAgent(Agent):
    def __init__(self, node_dim : int, action_dim : int, config : Config):
        super().__init__(config.target_update, config.update_frequency, config.learning_steps)
        self._memory = DynamicMemoryBuffer(buffer_length=config.buffer_size, 
                                           sample_size=config.sample_size)
        
        self._embed = Encoder(dim=config.hypervec_dim, 
                              node_dim=node_dim)
        
        self._q_func = QFunction(embed_dim=config.hypervec_dim, 
                                 action_dim=action_dim)
        
        self._policy = Actor(dim=config.hypervec_dim, 
                             action_dim=action_dim)
        
        self._q_target = QFunctionTarget(qfunction=self._q_func, 
                                         tau=config.tau)
        
        self._alpha = Alpha(start=config.target_entropy_start, 
                            end=config.target_entropy_end, 
                            midpoint=config.target_entropy_midpoint, 
                            slope=config.target_entropy_slope, 
                            max_steps=config.max_steps,
                            autotune=config.autotune, 
                            alpha_value=config.alpha_value)
        
        self._policy_optim = optim.Adam(self._policy.parameters(), lr=config.policy_lr)
        self._alpha_optim = optim.Adam(self._alpha.parameters(), lr=config.alpha_lr)
        
        self._action_dim = action_dim
        self._config = config

        if config.gpu:
            device = torch.device(f'cuda:{config.gpu_device}')
        else:
            device = torch.device('cpu')

        for obj in [self._q_embedding, self._policy_embedding, self._q_func, self._q_func_target, self._policy, self._alpha]:
            obj.to(device)

        
    def param_update(self) -> dict[str : float]:
        trans = self._memory.sample()
        
        cur_state, cur_batch_index = self._embed(trans.state, trans.state_index)
        _, cur_prob, cur_log_prob = self._policy.sample_action(cur_state, cur_batch_index, trans.state_index)
        
        number_devices = torch.diff(trans.state_index)
        cur_log_prob = cur_log_prob / torch.log(number_devices * self._action_dim).view(-1, 1)
        
        with torch.no_grad():
            cur_q1, cur_q2 = self._q_func(cur_state, cur_batch_index, trans.state_index)
            cur_q_target = self._q_target(cur_state, cur_batch_index, trans.state_index)
            
            next_state_embed, next_batch_index = self._embed(trans.next_state, trans.next_state_index)
            
            next_q_target = self._q_target(next_state_embed, next_batch_index, trans.next_state_index)
            _, next_prob, next_log_prob = self._policy.sample_action(next_state_embed, next_batch_index, trans.next_state_index)
            
            next_log_prob = next_log_prob / torch.log(number_devices * self._action_dim).view(-1, 1) #Normilize by the maximum possible entropy
            batch_size, cur_action_size = cur_prob.shape
            
            ent = -torch.bmm(cur_prob.view(batch_size, 1, cur_action_size),
                            cur_log_prob.view(batch_size, cur_action_size, 1)).mean()
        
        policy_loss = sac.policy_loss(cur_q_target, cur_prob, cur_log_prob, self._alpha()).mean().squeeze()
        q1_dif, q2_dif = sac.q_func_loss(cur_q1, 
                                        cur_q2,
                                        next_q_target,
                                        trans.action,
                                        next_prob,
                                        next_log_prob,
                                        trans.reward,
                                        self._alpha(),
                                        self._config.discount,
                                        trans.done)
        
        #Need to get the vector corresponding to the chosen action which would involve doing integer divison between the action index and the total number of devices in a state    
        #Find what devices correspond to the choosen action and then select only the device embedding for that action
        
        with torch.no_grad():
            device_choosen = trans.action.squeeze() // self._action_dim
            indexes = trans.state_index[:-1] + device_choosen
            
            matrix_l1 = q1_dif * cur_state[indexes] * self._config.critic_lr
            matrix_l2 = q2_dif * cur_state[indexes] * self._config.critic_lr
            
            q1_params : Tensor
            q2_params : Tensor
            q1_params, q2_params = self._q_func.parameters()

            #Need to mod due to only a single model but multiple devices
            q1_params.index_add_(0, trans.action.squeeze() % self._action_dim, matrix_l1)
            q2_params.index_add_(0, trans.action.squeeze() % self._action_dim, matrix_l2)

            q1_loss = sac.mse(q1_dif)
            q2_loss = sac.mse(q2_dif)
        
        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()

        if self._config.autotune:
            alpha_loss = sac.alpha_loss(cur_prob,
                            cur_log_prob,
                            self._alpha(),
                            self._alpha.sigmoid_target_entropy())
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
        else:
            alpha_loss = torch.tensor(0, device='cpu')
        
        return {
            'Q1 Loss' : q1_loss.item(),
            'Q2 Loss' : q2_loss.item(),
            'Policy Loss' : policy_loss.item(),
            'Alpha Loss' : alpha_loss.item(),
            'Entropy' : ent.item(),
            'Alpha Value' : self._alpha().item()
        }
    
    def target_param_update(self):
        self._q_target.update()

    def sample(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            state_index = torch.tensor([0,state.shape[0]])
            embed_state, batch_index = self._embed(state, state_index)
            action, _, _ = self._policy.sample_action(embed_state, batch_index, state_index)
            return action
        
    def evaluate(self, state : Tensor) -> Tensor:
        with torch.no_grad():
            state_index = torch.tensor([0,state.shape[0]])
            embed_state, batch_index = self._embed(state, state_index)
            action = self._policy.evaluate_action(embed_state, batch_index, state_index)
            return action
        
    def add_data(self, trans : Transition) -> None:
        self._memory.add_data(trans)

    def save(self, directory : str):
        path = Path(directory) / 'policy.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._policy.state_dict(), path)