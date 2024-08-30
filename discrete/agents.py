from typing import Callable

from torch import Tensor
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.data_collection import MemoryBuffer, Transition
from . import nn, hdc


def create_nn_agent(input_size : int,
                 output_size : int,
                 hidden_size : int,
                 policy_lr : float,
                 critic_lr : float,
                 alpha_lr : float,
                 discount : float,
                 tau : float,
                 alpha_scale : float,
                 target_update : int, #When the target should update
                 update_frequency : int, #When the models should update,
                 summary_writer : SummaryWriter,
                 learning_steps : int, #Amount of gradient steps
                 device : torch.device):
    """Will create SAC agent based on NNs"""
    
    target_q = nn.QFunctionTarget(None, tau)
    alpha = nn.Alpha(output_size, alpha_scale, alpha_lr)

    actor = nn.Actor(input_size, 
                            output_size, 
                            hidden_size,
                            target_q,
                            alpha, 
                            policy_lr)
        
    q_function = nn.QFunction(input_size,
                                     output_size,
                                     hidden_size,
                                     actor,
                                     target_q,
                                     alpha,
                                     critic_lr,
                                     discount)
        
    target_q.set_actual(q_function)
    
    def update(trans : Transition) -> Tensor:
        """Will return tensor of the losses in a tensor of dim 6 in order of Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
        q_info = q_function.update(trans)
        actor_info = actor.update(trans)
        return torch.cat((q_info, actor_info))
    
    def call(state : Tensor) -> Tensor:
        """Will return the action that should be executed at the given state"""
        with torch.no_grad():
            action, _, _ = actor(state)
            return action
        
    #Moves to device 
    for obj in [target_q, alpha, actor, q_function]:
        obj.to(device)
    
    return Agent(
        target_update,
        update_frequency,
        summary_writer,
        learning_steps,
        update,
        call,
        target_q.update,
        actor.save
    )
    
def create_hdc_agent(input_size : int,
                 output_size : int,
                 hyper_dim : int,
                 policy_lr : float,
                 critic_lr : float,
                 discount : float,
                 tau : float,
                 alpha_scale : float,
                 target_update : int, #When the target should update
                 update_frequency : int, #When the models should update,
                 summary_writer : SummaryWriter,
                 learning_steps : int, #Amount of gradient steps
                 device : torch.device):
    """Will create SAC agent based on HDC"""
    
    actor_encoder = hdc.RBFEncoder(input_size, hyper_dim)
    critic_encoder = hdc.EXPEncoder(input_size, hyper_dim)

    target_q = hdc.TargetQFunction(tau, None)
    alpha = hdc.Alpha(output_size, alpha_scale, critic_lr)

    actor = hdc.Actor(hyper_dim,
                        output_size,
                        policy_lr,
                        actor_encoder,
                        alpha,
                        target_q)
    
    q_function = hdc.QFunction(hyper_dim,
                                    output_size,
                                    actor_encoder,
                                    critic_encoder,
                                    actor,
                                    target_q,
                                    alpha,
                                    critic_lr,
                                    discount)
    
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
            ae_state = actor_encoder(state)
            action, _, _ = actor(ae_state)
            return action
        
    return Agent(
        target_update,
        update_frequency,
        summary_writer,
        learning_steps,
        update,
        call,
        target_q.update,
        actor.save
    )
        
    

    
class Agent:
    
    def __init__(self,
                 target_update : int,
                 update_frequency : int,
                 summary_writer : SummaryWriter,
                 learning_steps : int,
                 update_func : Callable[[Transition], Tensor],
                 action_func : Callable[[Tensor], Tensor],
                 target_update_func : Callable[[], None],
                 save_actor_func : Callable[[str], None]):
        
        self._learning_steps = learning_steps
        self._summary_writer = summary_writer
        self._update_frequency = update_frequency
        self._target_update = target_update
        
        self._update_func = update_func
        self._action_func = action_func
        self._target_upate_func = target_update_func
        self._save_actor_func = save_actor_func
        
    def __call__(self, state : Tensor) -> Tensor:
        return self._action_func(state)
    
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
        #Critic
        self._summary_writer.add_scalar('QFunc1 Loss', logging_info[0], steps)
        self._summary_writer.add_scalar('QFunc2 Loss', logging_info[1], steps)
        
        #Alpha/Actor
        self._summary_writer.add_scalar('Actor Loss', logging_info[2], steps)
        self._summary_writer.add_scalar('Entropy', logging_info[3], steps)
        self._summary_writer.add_scalar('Alpha Loss', logging_info[4], steps)
        self._summary_writer.add_scalar('Alpha Value', logging_info[5], steps)
    