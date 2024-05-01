from torch import Tensor
import torch
from torch.utils.tensorboard import SummaryWriter

from .nn_implementation import QFunctionTarget, QFunction, Actor, Alpha
from data_collection import Transition

class Agent:

    def __init__(self, 
                 input_size : int,
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
                 summary_writer : SummaryWriter
                 ) -> None:
        
        self._target_q = QFunctionTarget(None, tau)
        self._alpha = Alpha(output_size, alpha_scale, alpha_lr)

        self._actor = Actor(input_size, 
                            output_size, 
                            hidden_size,
                            self._target_q,
                            self._alpha, 
                            policy_lr)
        
        self._q_function = QFunction(input_size,
                                     output_size,
                                     hidden_size,
                                     self._actor,
                                     self._target_q,
                                     self._alpha,
                                     critic_lr,
                                     discount)
        
        self._target_q.set_actual(self._q_function)
        
        self._target_update = target_update
        self._update_freq = update_frequency

        self._sw = summary_writer

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the action that should be executed at the given state"""
        with torch.no_grad():
            action, _, _ = self._actor(state)
            return action

    def update(self, batch : Transition, steps : int) -> None:
        """Will update the networks according to the correct steps"""
        if steps % self._update_freq == 0:
            self._q_function.update(batch, steps, self._sw)
            self._actor.update(batch, steps, self._sw)

        if steps % self._target_update == 0:
            self._target_q.update()

    def save_actor(self, extra_info : str = '') -> None:
        """Will save the actor to file named bestweights with extra_info concatenated to the end"""
        self._actor.save(f'bestweights_{extra_info}.pt')

    def to(self, device) -> None:
        """Moves agents assets to the device"""
        self._q_function.to(device)
        self._actor.to(device)
        self._target_q.to(device)
        self._alpha.to(device)