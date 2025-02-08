from typing import Callable
import random

from torch import Tensor, tensor
import torch
from torch.nn import utils

from utils import MemoryBuffer, Transition, LearningLogger, MAX_ROWS, DynamicMemoryBuffer
from . import nn, hdc, mil_nn, sac


def create_mil_nn_agent(device_size : int,
                 action_size : int,
                 embed_size : int,
                 pos_encode_size : int,
                 policy_lr : float,
                 critic_lr : float,
                 alpha_lr : float,
                 discount : float,
                 tau : float,
                 alpha_value : float,
                 target_update : int, #When the target should update
                 update_frequency : int, #When the models should update,
                 learning_steps : int, #Amount of gradient steps
                 device : torch.device,
                 buffer_length : int,
                 sample_size : int,
                 clip_norm_value : float,
                 random : random):
    
    embedding = mil_nn.Embedding(embed_size, pos_encode_size, device_size)
    q_func = mil_nn.QFunction(embed_size, action_size)
    q_func_target = mil_nn.QFunctionTarget(q_func, tau)
    policy = mil_nn.Actor(embed_size, action_size)
    
    optim = torch.optim.Adam([*embedding.parameters(), *q_func.parameters(), *policy.parameters()], lr=policy_lr)
    memory = DynamicMemoryBuffer(buffer_length, sample_size, random)
    
    for obj in [embedding, q_func, q_func_target, policy]:
        obj.to(device)
    
    def update(steps) -> Tensor:
        """Will return tensor of the losses in a tensor of dim 6 in order of Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
        
        trans = memory.sample()
        
        cur_state_embed, cur_batch_index = embedding(trans.state, trans.state_index)
        cur_q1, cur_q2 = q_func(cur_state_embed, cur_batch_index, trans.state_index)
        _, cur_prob, cur_log_prob = policy.sample_action(cur_state_embed, cur_batch_index)
        number_devices = torch.diff(trans.state_index)
        cur_log_prob = cur_log_prob / torch.log(number_devices * action_size).view(-1, 1) #Normilize by the maximum possible entropy
        with torch.no_grad():
            cur_q_target = q_func_target(cur_state_embed, cur_batch_index, trans.state_index)
            
            next_state_embed, next_batch_index = embedding(trans.next_state, trans.next_state_index)
            next_q_target = q_func_target(next_state_embed, next_batch_index, trans.next_state_index)
            _, next_prob, next_log_prob = policy.sample_action(next_state_embed, next_batch_index)
            next_log_prob = next_log_prob / torch.log(number_devices * action_size).view(-1, 1) 
            batch_size, cur_action_size = cur_prob.shape
            
            ent = -torch.bmm(cur_prob.view(batch_size, 1, cur_action_size),
                            cur_log_prob.view(batch_size, cur_action_size, 1)).mean()
        
        #Normilize entropy
        
        policy_loss = sac.policy_loss(cur_q_target, cur_prob, cur_log_prob, alpha_value).mean().squeeze()
        q1_dif, q2_dif = sac.q_func_loss(cur_q1, 
                                         cur_q2,
                                         next_q_target,
                                         trans.action,
                                         next_prob,
                                         next_log_prob,
                                         trans.reward,
                                         alpha_value,
                                         discount,
                                         trans.done)
        q1_loss = sac.mse(q1_dif)
        q2_loss = sac.mse(q2_dif)
        
        loss = policy_loss + q1_loss + q2_loss
        
        optim.zero_grad()
        loss.backward()
        
        utils.clip_grad_norm_(q_func._q1.parameters(), clip_norm_value)
        utils.clip_grad_norm_(q_func._q2.parameters(), clip_norm_value)
        
        optim.step()
        
        
        return torch.tensor([
            q1_loss,
            q2_loss,
            policy_loss,
            ent,
            0,
            alpha_value
        ]).to(device)
    
    def call(state : Tensor) -> Tensor:
        with torch.no_grad():
            embed_state, batch_index = embedding(state, torch.tensor([0,state.shape[0]]))
            action, _, _ = policy.sample_action(embed_state, batch_index)
            return action
        
    def evaluate(state : Tensor) -> Tensor:
        with torch.no_grad():
            embed_state, batch_index = embedding(state, torch.tensor([0, state.shape[0]]))
            action = policy.evaluate_action(embed_state, batch_index)
            return action
        
    def add_data(trans : Transition) -> None:
        memory.add_data(trans)
        
    return Agent(
        target_update,
        update_frequency,
        learning_steps,
        update,
        call,
        q_func_target.update,
        lambda x : None,
        evaluate,
        add_data
    )
            

def create_nn_agent(input_size : int,
                 output_size : int,
                 hidden_size : int,
                 policy_lr : float,
                 critic_lr : float,
                 alpha_lr : float,
                 discount : float,
                 tau : float,
                 alpha_value : float,
                 autotune : bool,
                 target_update : int, #When the target should update
                 update_frequency : int, #When the models should update,
                 learning_steps : int, #Amount of gradient steps
                 device : torch.device,
                 dynamic : bool,
                 grad_clip : float):
    """Will create SAC agent based on NNs"""
    
    if dynamic:
        actual_input_size = input_size * MAX_ROWS
    
    target_q = nn.QFunctionTarget(None, tau)
    alpha = nn.Alpha(output_size, alpha_value, alpha_lr, autotune=autotune)

    actor = nn.Actor(input_size, 
                            output_size, 
                            hidden_size,
                            target_q,
                            alpha, 
                            policy_lr,
                            dynamic,
                            grad_clip)
        
    q_function = nn.QFunction(input_size,
                                     output_size,
                                     hidden_size,
                                     actor,
                                     target_q,
                                     alpha,
                                     critic_lr,
                                     discount,
                                     dynamic,
                                     grad_clip)
        
    target_q.set_actual(q_function)
    
    for obj in [target_q, alpha, actor, q_function]:
        obj.to(device)
    
    def update(trans : Transition) -> Tensor:
        """Will return tensor of the losses in a tensor of dim 6 in order of Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
        if dynamic:
            trans = Transition(state = nn.pad(trans.state, actual_input_size),
                       next_state = nn.pad(trans.next_state, actual_input_size),
                       action=trans.action,
                       reward=trans.reward,
                       done=trans.done,
                       num_devices=trans.num_devices,
                       num_devices_n=trans.num_devices_n)
        q_info = q_function.update(trans)
        actor_info = actor.update(trans)
        return torch.cat((q_info, actor_info))
    
    def call(state : Tensor) -> Tensor:
        """Will return the action that should be executed at the given state"""
        with torch.no_grad():
            if dynamic: 
                actual_state = nn.pad(state, actual_input_size)
            else:
                actual_state = state
            #Even though this looks like it would work with dynamic False I do not think it would. Should just assume it is dynamic for this branch tbh
            action, _, _ = actor(actual_state, num_devices = tensor(state.shape[0]).unsqueeze(dim=0), batch_size = 1)
            return action
    
    def evaluate(state : Tensor) -> Tensor:
        with torch.no_grad():
            if dynamic: 
                actual_state = nn.pad(state, actual_input_size)
            else:
                actual_state = state
            return actor.evaluate(actual_state, tensor([state.shape[0]]))
    
    return Agent(
        target_update,
        update_frequency,
        learning_steps,
        update,
        call,
        target_q.update,
        actor.save,
        evaluate
    )
    
def create_hdc_agent(input_size : int,
                 output_size : int,
                 hyper_dim : int,
                 policy_lr : float,
                 critic_lr : float,
                 discount : float,
                 tau : float,
                 alpha_value : float,
                 autotune : bool,
                 target_update : int, #When the target should update
                 update_frequency : int, #When the models should update,
                 learning_steps : int, #Amount of gradient steps
                 device : torch.device,
                 dynamic : bool):
    """Will create SAC agent based on HDC"""
    
    actor_encoder = hdc.RBFEncoderFlatten(input_size, hyper_dim, dynamic)
    critic_encoder = hdc.EXPEncoderFlatten(input_size, hyper_dim, dynamic)

    target_q = hdc.TargetQFunction(tau, None)
    alpha = hdc.Alpha(output_size, alpha_value, critic_lr, autotune=autotune)

    actor = hdc.Actor(hyper_dim,
                        output_size,
                        policy_lr,
                        actor_encoder,
                        alpha,
                        target_q,
                        dynamic)
    
    q_function = hdc.QFunction(hyper_dim,
                                    output_size,
                                    actor_encoder,
                                    critic_encoder,
                                    actor,
                                    target_q,
                                    alpha,
                                    critic_lr,
                                    discount,
                                    dynamic)
    
    target_q.set_actual(q_function)
    
    for obj in [target_q, alpha, actor, q_function, actor_encoder, critic_encoder]:
        obj.to(device)
    
    def update(trans : Transition) -> Tensor:
        """Will update following SAC equations for HDC and return losses in the form of a six dimension tensor in the form
        Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
        
        ce_state, q_info = q_function.update(trans)
        actor_info = actor.update(trans, ce_state)
        return torch.cat((q_info, actor_info))
        
    
    def call(state : Tensor) -> Tensor:
        """Will use HDC actor to determine action that should be taken at the given state"""
        with torch.no_grad():
            ae_state = actor_encoder(state.squeeze())
            action, _, _ = actor(ae_state, num_devices = tensor(state.shape[0]).unsqueeze(dim=0), batch_size = 1)
            return action
        
    def evaluate(state : Tensor) -> Tensor:
        with torch.no_grad():
            ae_state = actor_encoder(state.flatten())
            return actor.evaluate(ae_state)
        
    return Agent(
        target_update,
        update_frequency,
        learning_steps,
        update,
        call,
        target_q.update,
        actor.save,
        evaluate
    )
        
    

    
class Agent:
    
    def __init__(self,
                 target_update : int,
                 update_frequency : int,
                 learning_steps : int,
                 update_func : Callable[[Transition], Tensor],
                 action_func : Callable[[Tensor], Tensor],
                 target_update_func : Callable[[], None],
                 save_actor_func : Callable[[str], None],
                 evaluate_func : Callable[[Tensor], Tensor],
                 add_data : Callable[[Transition], None]):
        
        self._learning_steps = learning_steps
        self._update_frequency = update_frequency
        self._target_update = target_update
        
        self._update_func = update_func
        self._action_func = action_func
        self._target_upate_func = target_update_func
        self._save_actor_func = save_actor_func
        self._evaluate_func = evaluate_func
        
        self._add_data = add_data
        
    def __call__(self, state : Tensor) -> Tensor:
        return self._action_func(state)
    
    def evaluate(self, state : Tensor) -> Tensor:
        return self._evaluate_func(state)
    
    def update(self, steps : int) -> None:
        """Will perform the approaite update for the agent given the specific amount of steps"""
        if steps % self._update_frequency == 0:
            
            logging_info = torch.zeros(6)
            
            for _ in range(self._learning_steps):
                logging_info += self._update_func(steps)
                
            self._log_data(logging_info / self._learning_steps, steps)
            
        if steps % self._target_update == 0:
            self._target_upate_func()
            
    def save_actor(self, extension : str) -> None:
        """Will save the actor weights with the given extension"""
        self._save_actor_func(f'actor_weights{extension}.pt')
        
    def add_data(self, trans : Transition) -> None:
        """Will save the transition to memory"""
        self._add_data(trans)

    def _log_data(self, logging_info : Tensor, steps : int) -> None:
        """Will log the training data"""
        
        logger = LearningLogger()
        
        log_dict = {
            'QFunc1 Loss' : logging_info[0].item(),
            'QFunc2 Loss' : logging_info[1].item(),
            'Actor Loss' : logging_info[2].item(),
            'Entropy' : logging_info[3].item(),
            'Alpha Loss' : logging_info[4].item(),
            'Alpha Value' : logging_info[5].item()
        }
        
        logger.log_scalars(log_dict, steps=steps)
    