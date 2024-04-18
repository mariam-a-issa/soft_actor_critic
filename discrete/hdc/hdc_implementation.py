from copy import deepcopy

from torch import nn, Tensor, device, optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import torch.nn.functional as F

from encoders import RBFEncoder, EXPEncoder
from data_collection import Transition

_EPS = 1e-4

#Copied from nn implementation could there be another way to do this?
class Alpha:

    def __init__(self, 
                action_space_size : int,
                scale : float,
                lr : float,
                dev : device) -> None:
        
        self._target_ent = -scale * torch.log(1 / torch.tensor(action_space_size))
        self._log_alpha = torch.zeros(1, requires_grad=True)
        self._optim = optim.Adam([self._log_alpha], lr = lr, eps=_EPS)

        self._target_ent.to(dev)
        self._log_alpha.to(dev)

    def __call__(self) -> float:
        """Will give the current alpha"""
        return self._log_alpha.exp().item()
    
    def update(self, log_probs : Tensor, action_probs : Tensor, steps : int, summary_writer : SummaryWriter) -> None:
        """Will update according to equation 11"""
        loss = (action_probs.detach() * (-self._log_alpha.exp() * (log_probs + self._target_ent).detach())).mean()
    
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        summary_writer.add_scalar('Alpha Loss', loss, steps)

class QModel:

    def __init__(self, hvec_dim : int, action_dim : int, dev : device) -> None:
        """Will create a model that is a matrix that contains a hypervector for each action"""
        upper_bound = 1 / torch.sqrt(hvec_dim)
        lower_bound = -upper_bound
        
        #Using the same initilzation as the torch.nn.Linear 
        #https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L106-L108

        self._model = (upper_bound - lower_bound) * torch.rand(action_dim, hvec_dim, device = dev) + lower_bound
        self._hdvec_dim = hvec_dim

    def __call__(self, state : Tensor) -> Tensor:
        """Parameter is a batch of encoded states and will 
        return the batch of vectors where each element is the actions q value
        
        b x hd -> b x a

        """

        return torch.real(torch.matmul(torch.conj(self._model), state.unsqueeze(dim = 2)).squeeze() / self._hdvec_dim)
    
    def parameters(self) -> Tensor:
        """Will return the hypervector model"""
        return self._model
    

class QFunction:

    def __init__(self, hvec_dim : int, 
                 action_dim : int, 
                 dev : device, 
                 encoder : RBFEncoder, 
                 actor : 'Actor',
                 target : 'TargetQFunction',
                 alpha : Alpha,
                 lr : float,
                 discount : float) -> None:
        """Will create a Q function that has two q models"""

        self._q1 = QModel(hvec_dim, action_dim, dev)
        self._q2 = QModel(hvec_dim, action_dim, dev)

        self.encoder = encoder

        self._actor = actor
        self._target = target
        self._lr = lr
        self._discount = discount
        self._alpha = alpha

    def __call__(self, state) -> Tensor:
        """State should be an encoded h_vect"""
        return torch.min(self._q1(state), self._q2(state))
    
    def update(self, trans : Transition, steps : int, summary_writer : SummaryWriter) -> None:
        """Use equation 10 to find loss and then bind loss"""

        #Start of copied code from nn_implementation
        with torch.no_grad():
            _, next_log_pi, next_action_probs = self._actor(trans.next_state)
            q_log_dif : Tensor = self._target(trans.next_state) - self._alpha() * next_log_pi

            #Unsqueeze in order to have b x 1 x a · b x a x 1
            #Which results in b x 1 x 1 to then be squeezed to b x 1 

            next_v = torch.bmm(next_action_probs.unsqueeze(dim=1), q_log_dif.unsqueeze(dim=-1)).squeeze(dim=-1)

            next_q = trans.reward + (1 - trans.done) * self._discount * next_v

        q1 : Tensor = self._q1(trans.state)
        q2 : Tensor = self._q2(trans.state)

        #The action will be b x 1 where each element corresponds to index of action
        #By doing gather, make q_a with shape b x 1 where the element is the q value for the performed action
        
        q1_a = q1.gather(1, trans.action)
        q2_a = q2.gather(1, trans.action)
        #Stop of copy

        #Issue
        #Currently can have a vector representing the loss for each element of the batch and 
        #can have vector where each element represents the index of thea action taken
        #Need to make it so that the mean of the losses for a given action update the specific hypervector that the action is attached to
        #Potentially have a matrix that has a one at the index of the action, use that to get a vector of the sum of losses for each action 
        #Then have another vector representing the amount of times the action is taken and then do component wise division
        

class TargetQFunction:
    
    def __init__(self,
                 tau : int,
                 encoder : RBFEncoder, 
                 q_function : QFunction) -> None:
        
        self.encoder = encoder

        self._actual = q_function

        self._q1 = deepcopy(q_function._q1)
        self._q2 = deepcopy(q_function._q2)

        self._tau = tau
    
    def __call__(self, state) -> Tensor:
        return torch.min(self._q1(state), self._q2(state))

    def update(self) -> None:
        """Will do polyak averaging to each model in the target"""
        for param, target_param in zip(self._actual._q1.parameters(), self._q1.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._actual._q2.parameters(), self._q2.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)


class Actor(nn.Module):

    def __init__(self, 
                 hvec_dim : int, 
                 action_dim : int, 
                 dev : device,
                 lr : int, 
                 encoder : EXPEncoder,
                 alpha : Alpha, 
                 target_q : TargetQFunction) -> None:
        super().__init__()

        self.encoder = encoder
        
        self._logits = nn.Linear(hvec_dim, action_dim, bias=False)
        
        self._target_q_function = target_q
        self._alpha = alpha

        self._optim = optim.Adam(self.parameters(), lr=lr)

        self._logits.to(dev)

    def forward(self, state : Tensor) -> tuple[Tensor]:
        """Will give the action, log_prob, action_probs of action"""

        #Basically the same as the nn_implementation
        logits = self._logits(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_probs = dist.probs
        log_prob = F.log_softmax(logits, dim=-1)

        return action, log_prob, action_probs
    
    def update(self, trans : Transition, steps : int, summary_writer : SummaryWriter):
        """Using according to equation 12 as well as gradient based """
        
        #Exact same as the nn_implementation as it is doing gradient
    
        _, log_probs, action_probs = self(trans.state)

        with torch.no_grad():
            q_v = self._target(trans.state)
        
        difference = self._alpha() * log_probs - q_v

        loss : Tensor = torch.bmm(action_probs.unsqueeze(dim=1), difference.unsqueeze(dim=-1)).mean() #Don't need squeeze as the result is b x 1 x 1 and mean will handle correctly

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        self._alpha.update(log_probs, action_probs, steps, summary_writer) #Do the update in the actor in order to not recaluate probs

        summary_writer.add_scalar('Actor Loss', loss, steps)



