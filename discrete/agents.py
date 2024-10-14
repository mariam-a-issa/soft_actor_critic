from typing import Callable

from torch import Tensor, tensor
import torch

from utils import MemoryBuffer, Transition, LearningLogger, MAX_ROWS
from . import nn, hdc

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
                 dynamic : bool):
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
                            dynamic)
        
    q_function = nn.QFunction(input_size,
                                     output_size,
                                     hidden_size,
                                     actor,
                                     target_q,
                                     alpha,
                                     critic_lr,
                                     discount,
                                     dynamic)
        
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
            return actor.evaluate(state)
    
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
    
    actor_encoder = hdc.RBFEncoder(input_size, hyper_dim, dynamic)
    critic_encoder = hdc.EXPEncoder(input_size, hyper_dim, dynamic)

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
                 evaluate_func : Callable[[Tensor], Tensor]):
        
        self._learning_steps = learning_steps
        self._update_frequency = update_frequency
        self._target_update = target_update
        
        self._update_func = update_func
        self._action_func = action_func
        self._target_upate_func = target_update_func
        self._save_actor_func = save_actor_func
        self._evaluate_func = evaluate_func
        
    def __call__(self, state : Tensor) -> Tensor:
        return self._action_func(state)
    
    def evaluate(self, state : Tensor) -> Tensor:
        return self._evaluate_func(state)
    
    def update(self, buffer : MemoryBuffer, steps : int) -> None:
        """Will perform the approaite update for the agent given the specific amount of steps"""
        if steps % self._update_frequency == 0:
            
            logging_info = torch.zeros(6)
            
            for _ in range(self._learning_steps):
                logging_info += self._update_func(buffer.sample())
                
            self._log_data(logging_info / self._learning_steps, steps)
            
        if steps % self._target_update == 0:
            self._target_upate_func()
            
    def save_actor(self, extension : str) -> None:
        """Will save the actor weights with the given extension"""
        self._save_actor_func(f'actor_weights{extension}.pt')

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
    