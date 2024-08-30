from copy import deepcopy
from pathlib import Path
import os
import math

from torch import nn, Tensor, optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import torch.nn.functional as F 

from .encoders import RBFEncoder, EXPEncoder
from utils.data_collection import Transition

_EPS = 1e-4

#Copied from nn implementation could there be another way to do this?
class Alpha:

    def __init__(self, 
                action_space_size : int,
                scale : float,
                lr : float) -> None:
        
        self._target_ent = -scale * torch.log(1 / torch.tensor(action_space_size))
        self._log_alpha = torch.zeros(1, requires_grad=True)
        self._optim = optim.Adam([self._log_alpha], lr = lr, eps=_EPS)
        self._action_s = action_space_size

    def __call__(self) -> Tensor:
        """Will give the current alpha"""
        return self._log_alpha.exp()
    
    def update(self, log_probs : Tensor, action_probs : Tensor, batch_size : int) -> tuple[Tensor, Tensor]:
        """Will update according to equation 11"""
        
        #Essentially batch dot product
        loss = torch.bmm(action_probs.detach().view(batch_size, 1, self._action_s).detach(), 
                        (-self._log_alpha.exp() * (log_probs + self._target_ent).detach()).view(batch_size, self._action_s, 1)).mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        
        return loss, self().squeeze() #Squeeze so that it is just the value


    def to(self, dev : torch.device) -> None:
        self._target_ent.to(dev)
        self._log_alpha.to(dev)

class QModel:

    def __init__(self, hvec_dim : int, action_dim : int) -> None:
        """Will create a model that is a matrix that contains a hypervector for each action"""
        upper_bound = 1 / math.sqrt(hvec_dim)
        lower_bound = -upper_bound
        
        #Using the same initilzation as the torch.nn.Linear 
        #https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L106-L108

        self._model = (upper_bound - lower_bound) * torch.rand(action_dim, hvec_dim, dtype=torch.cfloat) + lower_bound
        self._model.requires_grad_(False)
        self._hdvec_dim = hvec_dim
        self._action_dim = action_dim

    def __call__(self, state : Tensor) -> Tensor:
        """Parameter is a batch of encoded states and will 
        return the batch of vectors where each element is the actions q value
        
        b x hd -> b x a

        """
        
        # Need to broadcast model to batched state so state needs to be unsqueezed
        with torch.no_grad():
            return torch.real((torch.conj(self._model) @ state.unsqueeze(dim = 2)).squeeze() / self._hdvec_dim).view(state.shape[0], self._action_dim)
    
    def parameters(self) -> Tensor:
        return self._model
    
    def to(self, dev : torch.device) -> None:
        self._model.to(dev)
    

class QFunction:

    def __init__(self, hvec_dim : int, 
                 action_dim : int,
                 actor_encoder : RBFEncoder, 
                 critic_encoder : EXPEncoder, 
                 actor : 'Actor',
                 target : 'TargetQFunction',
                 alpha : Alpha,
                 lr : float,
                 discount : float) -> None:
        """Will create a Q function that has two q models"""

        self._q1 = QModel(hvec_dim, action_dim)
        self._q2 = QModel(hvec_dim, action_dim)

        self._a_encoder = actor_encoder
        self._c_encoder = critic_encoder

        self._actor = actor
        self._target = target
        self._lr = lr
        self._discount = discount
        self._alpha = alpha
        self._action_s = action_dim

    def __call__(self, state) -> Tensor:
        """State should be an encoded h_vect"""
        return torch.min(self._q1(state), self._q2(state))
    
    def update(self, trans : Transition) -> tuple[Tensor, Tensor]:
        """Use equation 10 to find loss and then bind loss
           Return the ce_state in order to not recalculate it and the loss"""
        
        batch_size = len(trans.state)

        #Start of copied code from nn_implementation
        with torch.no_grad():
            ae_next_state = self._a_encoder(trans.next_state)
            ce_next_state = self._c_encoder(trans.next_state)
            ce_state = self._c_encoder(trans.state)

            next_action_probs : Tensor
            _, next_log_pi, next_action_probs = self._actor(ae_next_state)
            q_log_dif : Tensor = self._target(ce_next_state) - self._alpha() * next_log_pi

 
            #Essentially batch dot product
            next_v = torch.bmm(next_action_probs.view(batch_size, 1, self._action_s), 
                               q_log_dif.view(batch_size, self._action_s, 1)).view(batch_size, 1)

            next_q = trans.reward + (1 - trans.done) * self._discount * next_v

            q1 : Tensor = self._q1(ce_state)
            q2 : Tensor = self._q2(ce_state)

            #The action will be b x 1 where each element corresponds to index of action
            #By doing gather, make q_a with shape b x 1 where the element is the q value for the performed action
            
            q1_a = q1.gather(1, trans.action)
            q2_a = q2.gather(1, trans.action)
            #Stop of copy

            l1 : Tensor = next_q - q1_a
            l2 : Tensor = next_q - q2_a

            #Creates a matrix where each row is the hypervector that should be bundled with the model
            matrix_l1 = l1 * ce_state * self._lr
            matrix_l2 = l2 * ce_state * self._lr

            #Index add will add the vector found at index i of matrix_l1 to index a_i of the model (returned by parameters()),
            #where a_i is the value of trans.action at index i
            #trans.action is a b x 1 column vector but needs to be row vector so squeeze
            self._q1.parameters().index_add_(0, trans.action.squeeze(), matrix_l1)
            self._q2.parameters().index_add_(0, trans.action.squeeze(), matrix_l2)
        
        return ce_state, torch.stack((1/2 * (l1 ** 2).mean(), 1/2 * (l2 ** 2).mean()))


    def to(self, device : torch.device) -> None:
        """Moves q function to device"""
        self._q1.to(device)
        self._q2.to(device)

class TargetQFunction:
    
    def __init__(self,
                 tau : int,
                 q_function : QFunction) -> None:

        self._actual = q_function

        if q_function is not None:
            self._q1 = deepcopy(q_function._q1)
            self._q2 = deepcopy(q_function._q2)

        self._tau = tau

    def set_actual(self, q_function : QFunction) -> None:
        """Will actually set the q_function if it was not set in init"""
        self._actual = q_function

        self._q1 = deepcopy(q_function._q1)
        self._q2 = deepcopy(q_function._q2)
    
    def __call__(self, state) -> Tensor:
        return torch.min(self._q1(state), self._q2(state))

    def update(self) -> None:
        """Will do polyak averaging to each model in the target"""
        for param, target_param in zip(self._actual._q1.parameters(), self._q1.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._actual._q2.parameters(), self._q2.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
    
    def to(self, dev : torch.device) -> None:
        self._q1.to(dev)
        self._q2.to(dev)

class Actor(nn.Module):

    def __init__(self, 
                 hvec_dim : int, 
                 action_dim : int, 
                 lr : int, 
                 actor_encoder : RBFEncoder,
                 alpha : Alpha, 
                 target_q : TargetQFunction) -> None:
        super().__init__()

        self._a_encoder = actor_encoder
        
        self._logits = nn.Linear(hvec_dim, action_dim, bias=False)
        self._logits.weight.data = torch.zeros((action_dim, hvec_dim))
        
        self._target = target_q
        self._alpha = alpha

        self._optim = optim.Adam(self.parameters(), lr=lr, eps = _EPS)

        self._action_s = action_dim

    def forward(self, state : Tensor) -> tuple[Tensor]:
        """Will give the action, log_prob, action_probs of action"""

        #Same as the nn_implementation
        logits = self._logits(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_probs = dist.probs
        log_prob = F.log_softmax(logits, dim=-1)

        return action, log_prob, action_probs
    
    def evaluate(self, state : Tensor) -> Tensor:
        """Will return the best action for evaulation"""
        
        return torch.argmax(self._logits(state))
    
    def update(self, trans : Transition, ce_state : Tensor) -> Tensor:
        """Using according to equation 12 as well as gradient based and return the actors loss, actors antropy, alpha loss, and the current alpha"""
        
        #Same as the nn_implementation as it is doing gradient

        batch_size = len(trans.state)
        
        ae_state = self._a_encoder(trans.state)
        q_v = self._target(ce_state)

        action_probs : Tensor; log_probs : Tensor; difference : Tensor ; loss : Tensor
        _, log_probs, action_probs = self(ae_state)
        
        difference = self._alpha() * log_probs - q_v

        #Essentially batch dot product
        loss = torch.bmm(action_probs.view(batch_size, 1, self._action_s),  difference.view(batch_size, self._action_s, 1)).mean()

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        alpha_loss, alpha = self._alpha.update(log_probs, action_probs, batch_size) #Do the update in the actor in order to not recaluate probs

        with torch.no_grad():
            ent = -torch.bmm(action_probs.view(batch_size, 1, self._action_s), log_probs.view(batch_size, self._action_s, 1)).mean()

            return torch.stack((loss, ent, alpha_loss, alpha))
        
    def save(self, file_name ='best_weights.pt') -> None:
        """Will save the model in the folder 'model' in the dir that the script was run in."""

        folder_name = type(self).__name__

        model_folder_path = Path('./model/' + folder_name)
        file_dir = Path(os.path.join(model_folder_path, file_name))

        if not os.path.exists(file_dir.parent):
            os.makedirs(file_dir.parent)

        torch.save(self.state_dict(), file_dir)



