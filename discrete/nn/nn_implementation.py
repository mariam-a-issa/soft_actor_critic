from copy import deepcopy

import torch
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .base_nn import BaseNN
from utils.data_collection import Transition

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
    
    def update(self, trans : Transition) -> Tensor:
        """Will update using equations 3, 4, and 12 and return the loss for both q functions"""
        
        batch_size = len(trans.state)
        
        with torch.no_grad():
            next_log_pi : Tensor
            next_action_probs : Tensor
            _, next_log_pi, next_action_probs = self._actor(trans.next_state)
            q_log_dif : Tensor = self._target(trans.next_state) - self._alpha() * next_log_pi
            
            #Batch wise dot product
            next_v = torch.bmm(next_action_probs.view(batch_size, 1, self._action_s), 
                               q_log_dif.view(batch_size, self._action_s, 1)).view(batch_size, 1)
            
            next_q : Tensor = trans.reward + (1 - trans.done) * self._discount * next_v

        q1 : Tensor = self._q1(trans.state)
        q2 : Tensor = self._q2(trans.state)

        #The action will be b x 1 where each element corresponds to index of action
        #By doing gather, make q_a with shape b x 1 where the element is the q value for the performed action
        
        q1_a = q1.gather(1, trans.action)
        q2_a = q2.gather(1, trans.action)

        self._optim1.zero_grad()
        self._optim2.zero_grad()

        ls1 = 1/2 * ((q1_a - next_q) ** 2).mean()
        ls2 = 1/2 * ((q2_a - next_q) ** 2).mean()

        ls1.backward()
        ls2.backward()
        
        self._optim1.step()
        self._optim2.step()
        
        return torch.stack((ls1, ls2))

    def to(self, device) -> None:
        """Will move the QFunction to the device"""
        self._q1.to(device)
        self._q2.to(device)
        
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

    def __call__(self) -> Tensor:
        """Will give the current alpha"""
        return self._log_alpha.exp()
    
    def update(self, log_probs : Tensor, action_probs : Tensor, batch_size : int) -> tuple[Tensor, Tensor]:
        """Will update according to equation 11"""
        #Batch wise dot prodcut then mean
        loss = torch.bmm(action_probs.detach().view(batch_size, 1, self._action_s), 
                         (-self._log_alpha.exp() * (log_probs + self._target_ent).detach()).view(batch_size, self._action_s, 1)).mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        
        return loss, self().squeeze() #Squeeze so that it is just the value


class Actor(BaseNN):

    def __init__(self, 
                 input_size: int,
                 output_size: int, 
                 hidden_size,
                 target : QFunctionTarget, 
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
    
    def evaluate(self, x : Tensor) -> Tensor:
        """Will return the best action for evaulation"""
        
        return torch.argmax(super().forward(x))
    
    def update(self, trans : Transition) -> Tensor:
        """Will update according to equation 12 and return the actors loss, actors entropy, alpha_loss, and the current alpha"""

        batch_size = len(trans.state)
        
        action_probs : Tensor; log_probs : Tensor; difference : Tensor; loss : Tensor
        _, log_probs, action_probs = self(trans.state)

        # with torch.no_grad():
        q_v = self._q_func(trans.state)
        
        difference = self._alpha() * log_probs - q_v

        loss = torch.bmm(action_probs.view(batch_size, 1, self._action_s),  difference.view(batch_size, self._action_s, 1)).mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        alpha_loss, alpha = self._alpha.update(log_probs, action_probs, batch_size) #Do the update in the actor in order to not recaluate probs

        with torch.no_grad():
            ent = -torch.bmm(action_probs.view(batch_size, 1, self._action_s), log_probs.view(batch_size, self._action_s, 1)).mean()
            
            return torch.stack((loss, ent, alpha_loss, alpha))
        
    def set_actual(self, q_func : QFunction) -> None:
        """Will actually set the q_func if it was not set in the constructor"""
        self._q_func = q_func
