from pathlib import Path
from copy import deepcopy

from torch import Tensor
from torch.nn import utils
import torch
from torch_geometric.data import Data, Batch

from utils import GraphMemoryBuffer, DynamicMemoryBuffer, Transition, Config, group_to_boundaries_torch
from .implementation import AttentionEmbedding, GraphEmbedding, Embedding, Actor, QFunction, QFunctionTarget
from ..agents import Agent
from .. import sac


class MILNNAgent(Agent):

    def __init__(self, node_dim : int, action_dim : int, config : Config):
        super().__init__(config.target_update, config.update_frequency, config.learning_steps)
    
    #When creating the embedding functions understand the output sizes of the embeddings.
        if config.attention:
            self._q_embedding = AttentionEmbedding(embed_dim=config.hidden_dim, 
                                                   pos_enc_dim=config.pos_enc_dim, 
                                                   node_dim=node_dim, 
                                                   num_heads=config.num_heads)
            
            self._target_q_embedding = deepcopy(self._q_embedding)

            self._policy_embedding = AttentionEmbedding(embed_dim=config.hidden_dim, 
                                                   pos_enc_dim=config.pos_enc_dim, 
                                                   node_dim=node_dim, 
                                                   num_heads=config.num_heads)
        elif config.graph:
            #Need to multiply by two here as we do not concat our global embedding as there is non in this model
            self._q_embedding = GraphEmbedding(embed_dim=config.hidden_dim * 2, 
                                               pos_enc_dim=config.pos_enc_dim,
                                               node_dim=node_dim, 
                                               message_passes=config.messages_passes)
            
            self._target_q_embedding = deepcopy(self._q_embedding)

            self._policy_embedding = GraphEmbedding(embed_dim=config.hidden_dim * 2, 
                                               pos_enc_dim=config.pos_enc_dim,
                                               node_dim=node_dim, 
                                               message_passes=config.messages_passes)
        else:
            self._q_embedding = Embedding(embed_dim=config.hidden_dim, 
                                          pos_enc_dim=config.pos_enc_dim, 
                                          node_dim=node_dim)
            
            self._target_q_embedding = deepcopy(self._q_embedding)

            self._policy_embedding = Embedding(embed_dim=config.hidden_dim, 
                                            pos_enc_dim=config.pos_enc_dim, 
                                            node_dim=node_dim)

        config.hidden_dim *= 2 #The hidden size doubles after the concatination of the global embedding    

        self._q_func = QFunction(embed_dim=config.hidden_dim, action_dim=action_dim)
        self._q_func_target = QFunctionTarget(self._q_func, tau=config.tau)
        self._policy = Actor(embed_dim=config.hidden_dim, action_dim=action_dim)
        self._alpha = sac.Alpha(start=config.target_entropy_start, 
                            end=config.target_entropy_end, 
                            midpoint=config.target_entropy_midpoint, 
                            slope=config.target_entropy_slope, 
                            max_steps=config.max_steps, 
                            autotune=config.autotune, 
                            alpha_value=config.alpha_value)

        self._optim_critic = torch.optim.Adam([*self._q_embedding.parameters(), *self._q_func.parameters()], lr=config.critic_lr)
        self._optim_policy = torch.optim.Adam([*self._policy_embedding.parameters(), *self._policy.parameters()], lr=config.policy_lr)
        self._optim_alpha = torch.optim.Adam([self._alpha._log_alpha], lr = config.alpha_lr)

        if config.graph:
            self._memory = GraphMemoryBuffer(config.buffer_size, config.sample_size)
        else:
            self._memory = DynamicMemoryBuffer(config.buffer_size, config.sample_size)
        
        if config.gpu:
            device = torch.device(f'cuda:{config.gpu_device}')
        else:
            device = torch.device('cpu')

        for obj in [self._q_embedding, self._policy_embedding, self._q_func, self._q_func_target, self._policy, self._alpha]:
            obj.to(device)

        self._action_dim = action_dim
        self._config = config
    
    def param_update(self) -> dict[str : float]:
        """Will return tensor of the losses in a tensor of dim 6 in order of Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
        
        trans = self._memory.sample()
        
        q_cur_state_embed, q_cur_batch_index = self._q_embedding(trans.state, trans.state_index)
        policy_cur_state_embed, policy_cur_batch_index = self._policy_embedding(trans.state, trans.state_index)
        
        cur_q1, cur_q2 = self._q_func(q_cur_state_embed, q_cur_batch_index, trans.state_index)
        _, cur_prob, cur_log_prob = self._policy.sample_action(policy_cur_state_embed, policy_cur_batch_index)
        
        number_devices = torch.diff(trans.state_index)
        cur_log_prob = cur_log_prob / torch.log(number_devices * self._action_dim).view(-1, 1) #Normilize by the maximum possible entropy
        
        with torch.no_grad():
            tar_q_cur_state_embed, tar_q_cur_batch_index = self._q_embedding(trans.state, trans.state_index)
            cur_q_target = self._q_func_target(tar_q_cur_state_embed, tar_q_cur_batch_index, trans.state_index)
            
            q_next_state_embed, q_next_batch_index = self._q_embedding(trans.next_state, trans.next_state_index)
            policy_next_state_embed, policy_next_batch_index = self._policy_embedding(trans.next_state, trans.next_state_index)
            
            next_q_target = self._q_func_target(q_next_state_embed, q_next_batch_index, trans.next_state_index)
            _, next_prob, next_log_prob = self._policy.sample_action(policy_next_state_embed, policy_next_batch_index)
            
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
        
        grad_policy = self.calc_grad_norm([*self._policy_embedding.parameters(), *self._policy.parameters()])
        grad_q_func = self.calc_grad_norm([*self._q_embedding.parameters(), *self._q_func.parameters()])

        self._optim_alpha.zero_grad()
        alpha_loss.backward()
        
        if self._config.grad_clip:
            utils.clip_grad_norm_([*self._q_embedding.parameters(), *self._q_func.parameters()], self._config.grad_clip)
        
        self._optim_policy.step()
        self._optim_critic.step()
        self._optim_alpha.step()
        

        return {
            'Q1 Loss' : q1_loss.item(),
            'Q2 Loss' : q2_loss.item(),
            'Policy Loss' : policy_loss.item(),
            'Alpha Loss' : alpha_loss.item(),
            'Entropy' : ent.item(),
            'Alpha Value' : self._alpha().item(),
            'Grad of Policy' : grad_policy,
            'Unclipped Grad of Q Func' : grad_q_func
        }
        
    def target_param_update(self):
        self._q_func_target.update()
        self.polyak_average(self._q_embedding.parameters(), self._target_q_embedding.parameters(), self._config.tau)
        
    def sample(self, state : Tensor | Data) -> Tensor:
        with torch.no_grad():
            
            if self._config.graph:
                state = Batch.from_data_list([state])
                state_index = group_to_boundaries_torch(state.batch)
            else:
                state_index = torch.tensor([0,state.shape[0]])
            
            embed_state, batch_index = self._policy_embedding(state, state_index)
            action, _, _ = self._policy.sample_action(embed_state, batch_index)
            return action
        
    def evaluate(self, state : Tensor | Data) -> Tensor:
        with torch.no_grad():
            
            if self._config.graph:
                state = Batch.from_data_list([state])
                state_index = group_to_boundaries_torch(state.batch)
            else:
                state_index = torch.tensor([0,state.shape[0]])
            
            embed_state, batch_index = self._policy_embedding(state, state_index)
            action = self._policy.evaluate_action(embed_state, batch_index)
            return action
        
    def add_data(self, trans : Transition) -> None:
        self._memory.add_data(trans)

    def save(self, directory : str):
        path = Path(directory) / 'policy.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._policy.state_dict(), path)