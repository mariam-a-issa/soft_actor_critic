from math import pi

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from .hdc_implementation import TargetQFunction, QFunction, Actor, Alpha
from .encoders import RBFEncoder, EXPEncoder
from data_collection import Transition

class Agent:

    def __init__(self, 
                 input_size : int,
                 output_size : int,
                 hyper_dim : int,
                 policy_lr : float,
                 critic_lr : float,
                 discount : float,
                 tau : float,
                 alpha_scale : float,
                 target_update : int, #When the target should update
                 update_frequency : int, #When the models should update,
                 summary_writer : SummaryWriter
                 ) -> None:
        
        
        self._actor_encoder = RBFEncoder(input_size, hyper_dim)
        self._critic_encoder = EXPEncoder(input_size, hyper_dim)

        self._target_q = TargetQFunction(tau, None)
        self._alpha = Alpha(output_size, alpha_scale, critic_lr)

        self._actor = Actor(hyper_dim,
                            output_size,
                            policy_lr,
                            self._actor_encoder,
                            self._critic_encoder,
                            self._alpha,
                            self._target_q)
        
        self._q_function = QFunction(hyper_dim,
                                     output_size,
                                     self._actor_encoder,
                                     self._critic_encoder,
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
            hvec_state = self._actor_encoder(state)
            action, _, _ = self._actor(hvec_state)
            return action

    def update(self, batch : Transition, steps : int) -> None:
        """Will update the networks according to the correct steps"""
        if steps % self._update_freq == 0:
            ce_state = self._q_function.update(batch, steps, self._sw)
            self._actor.update(batch, steps, self._sw, ce_state)

        if steps % self._target_update == 0:
            self._target_q.update()

    def save_actor(self, extension : str) -> None:
        """Will save the actor"""
        self._actor.save(f'bestweights{extension}.pt')

    def to(self, device) -> None:
        """Moves agents assets to the device"""
        self._q_function.to(device)
        self._actor.to(device)
        self._target_q.to(device)
        self._alpha.to(device)