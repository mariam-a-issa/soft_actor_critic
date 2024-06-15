from copy import deepcopy

import torch
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from .base_nn import BaseNN
from data_collection import Transition

import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

#Parameter update implementation from https://arxiv.org/abs/1910.07207

_EPS = 1e-4 #Term used in cleanrl

class QFunction:

    def __init__(self, 
                 input_size : int, 
                 output_size : int, 
                 hidden_size : int, 
                 actor : 'Actor', 
                 target : 'QFunctionTarget',
                 alpha : 'Alpha',
                 lr : float,
                 discount : float) -> None:
        
        """Will create a q function that will use two q models"""
        self._q1 = BaseNN(input_size, output_size, hidden_size, id=1)
        self._q2 = BaseNN(input_size, output_size, hidden_size, id=2)
        
        self._optim1 = optim.Adam(self._q1.parameters(), lr=lr, eps=_EPS)
        self._optim2 = optim.Adam(self._q2.parameters(), lr=lr, eps=_EPS)

        self._actor = actor
        self._target = target
        self._alpha = alpha
        self._discount = discount
        
        self._action_s = output_size


    def set_actor(self, actor : 'Actor') -> None:
        """Will set the actor used for parameter updates"""
        self._actor = actor

    def set_target(self, target : 'QFunction') -> None:
        self._target = target

    def __call__(self, state : Tensor) -> Tensor:
        """Will give a Tensor where each index represents the q value for the corresponding action"""
        return torch.min(self._q1(state), self._q2(state))
    
    def update(self, trans : Transition, steps : int, summary_writer : SummaryWriter) -> None:
        """Will update using equations 3, 4, and 12"""
        
        batch_size = len(trans.state)
        
        with torch.no_grad():
            next_log_pi : Tensor
            next_action_probs : Tensor
            _, next_log_pi, next_action_probs = self._actor(trans.next_state)
            q_log_dif : Tensor = self._target(trans.next_state) - self._alpha() * next_log_pi
            
            #Batch wise dot product
            next_v = torch.bmm(next_action_probs.view(batch_size, 1, self._action_s), 
                               q_log_dif.view(batch_size, self._action_s, 1)).view(batch_size, 1)
            
            next_q = trans.reward + (1 - trans.done) * self._discount * next_v

        q1 : Tensor = self._q1(trans.state)
        q2 : Tensor = self._q2(trans.state)

        #The action will be b x 1 where each element corresponds to index of action
        #By doing gather, make q_a with shape b x 1 where the element is the q value for the performed action
        
        q1_a = q1.gather(1, trans.action)
        q2_a = q2.gather(1, trans.action)

        self._optim1.zero_grad()
        self._optim2.zero_grad()

        self._calculate_losses(q1_a, q2_a, next_q, steps, summary_writer)

        #logging.debug(f"QFunction update step {steps}, q1_loss: {q1_a.mean().item()}, q2_loss: {q2_a.mean().item()}, next_q: {next_q.mean().item()}")
        self._optim1.step()
        self._optim2.step()

    def to(self, device) -> None:
        """Will move the QFunction to the device"""
        self._q1.to(device)
        self._q2.to(device)

    def _calculate_losses(self,
                          actual1 : Tensor, 
                          actual2 : Tensor, 
                          expected : Tensor, 
                          steps : int,
                          summary_writer : SummaryWriter) -> Tensor:
        
        """Will calculate the loss for both q according to equation 4 then backprop"""
        ls1 = 1/2 * ((actual1 - expected) ** 2).mean()
        ls2 = 1/2 * ((actual2 - expected) ** 2).mean()

        ls1.backward()
        ls2.backward()
        
        #clip_grad_norm_(self._q1.parameters(), 1)
        #clip_grad_norm_(self._q2.parameters(), 1)

        
        summary_writer.add_scalar('QFunc1 Loss', ls1, steps)
        summary_writer.add_scalar('QFunc2 Loss', ls2, steps)
        
class QFunctionTarget:

    def __init__(self, actual : QFunction, tau : float) -> None:
        self._actual = actual

        if actual is not None:
            self._target = deepcopy(actual)

        self._tau = tau

    def set_actual(self, actual : QFunction) -> None:
        """Will set the actual if it was not set in init"""
        self._actual = actual
        self._target = deepcopy(actual)

    def to(self, device) -> None:
        self._actual.to(device)
        self._target.to(device)

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the q values from the target network"""
        return self._target(state)
    
    def update(self) -> None:
        """Will do polyak averaging to update the target"""
        for param, target_param in zip(self._actual._q1.parameters(), self._target._q1.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._actual._q2.parameters(), self._target._q2.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

class Alpha:

    def __init__(self, 
                action_space_size : int,
                scale : float,
                lr : float) -> None:
        
        self._target_ent = -scale * torch.log(1 / torch.tensor(action_space_size))
        self._log_alpha = torch.zeros(1, requires_grad=True)
        self._optim = optim.Adam([self._log_alpha], lr = lr, eps=_EPS)
        self._action_s = action_space_size

    def to(self, device) -> None:
        """Will move the alpha to the device"""
        self._target_ent.to(device)
        self._log_alpha.to(device)

    def __call__(self) -> float:
        """Will give the current alpha"""
        return self._log_alpha.exp()
    
    def update(self, log_probs : Tensor, action_probs : Tensor, steps : int, batch_size : int, summary_writer : SummaryWriter) -> None:
        """Will update according to equation 11"""
        #Batch wise dot prodcut then mean
        loss = torch.bmm(action_probs.detach().view(batch_size, 1, self._action_s).detach(), 
                         (-self._log_alpha.exp() * (log_probs + self._target_ent).detach()).view(batch_size, self._action_s, 1)).mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        summary_writer.add_scalar('Alpha Loss', loss, steps)
        summary_writer.add_scalar('Current Alpha', self._log_alpha.exp(), steps)


class Actor(BaseNN):

    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 hidden_size,
                 target : QFunction, 
                 alpha : 'Alpha',
                 lr : float) -> None:
        
        super().__init__(input_size, output_size, hidden_size)
        self._q_func = target
        self._alpha = alpha
        self._optim = optim.Adam(self.parameters(), lr=lr, eps=_EPS)
        
        self._action_s = output_size

    def forward(self, x : Tensor) -> tuple[Tensor]:
        """Will give the action, log_prob, and action_probs of action"""

        #Implementation very similar to cleanrl
        logits = super().forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_probs = dist.probs
        log_prob = F.log_softmax(logits, dim = -1)

        return action, log_prob, action_probs
    
    def update(self, trans : Transition, steps : int, summary_writer : SummaryWriter) -> None:
        """Will update according to equation 12"""

        batch_size = len(trans.state)
        
        action_probs : Tensor
        _, log_probs, action_probs = self(trans.state)

        # with torch.no_grad():
        q_v = self._q_func(trans.state)
        
        difference : Tensor = self._alpha() * log_probs - q_v

        loss : Tensor = torch.bmm(action_probs.view(batch_size, 1, self._action_s),  difference.view(batch_size, self._action_s, 1)).mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        self._alpha.update(log_probs, action_probs, steps, batch_size, summary_writer) #Do the update in the actor in order to not recaluate probs

        #logging.debug(f"Actor update step {steps}, loss: {loss.item()}, log_probs: {log_probs}, action_probs: {action_probs}")
        summary_writer.add_scalar('Actor Loss', loss, steps)
        
    def set_actual(self, q_func : QFunction) -> None:
        """Will actually set the q_func if it was not set in the constructor"""
        self._q_func = q_func
