from copy import deepcopy
import math

import torch
from torch import tensor, nn, optim, Tensor
from torch_scatter import segment_coo
from torch_scatter.composite import scatter_log_softmax
import torch.nn.functional as F 
from torch.distributions import Categorical

from .architecture import positional_encoding
from utils import EPS, LearningLogger

class Embedding(nn.Module):
    
    def __init__(self, 
                emb_dim : int,
                pos_enc_dim : int, #This needs to be even
                device_dim : int) -> None:
        super().__init__()
        self._embeding = nn.Sequential(nn.Linear(device_dim + pos_enc_dim, emb_dim), nn.LeakyReLU())
        self._inner = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim), nn.LeakyReLU()) #2 * for both the mean and the max
        self._pos_enc_dim = pos_enc_dim
        
    def forward(self, states : Tensor, state_index : Tensor) -> tuple[Tensor, Tensor]:
        """Will encode and then embed each set of devices in the list using postional encoding, embedding layer, and concatiaton of an aggregation
            states: All of the device states in a flattened mxn where m is the total numner of devices and n 2 x embd dim
            state_index : An array representing the start index of each batch. The last index should be len of states as this indicates where the next batch should go
            
            return tensor of same of shape mx2*emb dim
                   batch index where each element corresponds to a device in states and represents what element of the batch it is
        """
        
        pos_index = torch.cat([torch.arange(start = 1, end = state_index[i + 1] - state_index[i] + 1) for i in range(len(state_index) - 1)])

        pos_enc = positional_encoding(pos_index, self._pos_enc_dim)
        
        states = torch.cat((states, pos_enc), dim = 1)
        states = self._embeding(states)
        
        batch_index = torch.cat([torch.zeros(state_index[i + 1] - state_index[i], dtype=int) + i for i in range(len(state_index) - 1)]) #Will create an index that can be used by torch_scatter to reduce corresponding elements
        #TODO switch to pointer version of segment as segment_coo is non deterministic
        states_agg = torch.cat([segment_coo(states, batch_index, reduce='mean'), segment_coo(states, batch_index, reduce='max')], dim=1)
        states_agg = self._inner(states_agg)
        
        return torch.cat([states, states_agg[batch_index]], dim = 1), batch_index
    
class AttentionEmbedding(nn.Module):
    
    def __init__(self,
                 emb_dim : int,
                 pos_enc_dim : int,
                 device_dim : int,
                 num_heads : int) -> None:
        super().__init__()
        self._emb_dim = emb_dim
        self._pos_enc_dim = pos_enc_dim
        self._embedding = nn.Sequential(nn.Linear(device_dim + pos_enc_dim, self._emb_dim), nn.LeakyReLU())
        self._mha = nn.MultiheadAttention(self._emb_dim, num_heads, batch_first=True)
        self._norm = nn.LayerNorm(self._emb_dim)
        
    def forward(self, states : Tensor, state_index : Tensor) -> tuple[Tensor, Tensor]:
        batch_index = torch.cat([torch.zeros(state_index[i + 1] - state_index[i], dtype=int) + i for i in range(len(state_index) - 1)])
        
        pos_index = torch.cat([torch.arange(start = 1, end = state_index[i + 1] - state_index[i] + 1) for i in range(len(state_index) - 1)])

        pos_enc = positional_encoding(pos_index, self._pos_enc_dim)
        
        states = torch.cat((states, pos_enc), dim = 1)
        embed_states = self._embedding(states)
        
        reshape_embed_states = _reshape(embed_states, batch_index, self._emb_dim, filler_val=0).view(torch.unique(batch_index).numel(), -1, self._emb_dim) #batch_size x seq_length x embd size
        attn_output, _ = self._mha(reshape_embed_states, reshape_embed_states, reshape_embed_states)
        residual_output = attn_output + reshape_embed_states
        norm = self._norm(residual_output)

        # Count occurrences of each batch index
        _, counts = batch_index.unique(return_counts=True)

        # Create indices for each batch element
        seq_indices = torch.cat([torch.arange(n) for n in counts])

        # Gather the required elements
        selected_features = norm[batch_index, seq_indices, :]
        
        return torch.cat((embed_states, selected_features.view(-1, self._emb_dim)), dim=1), batch_index
        
        
class Alpha:

    def __init__(self,
                 start : float,
                 end : float,
                 midpoint : float,
                 slope : float,
                 max_steps : int,
                 auto_tune : bool = False,
                 alpha_value : float = None):
        """
        Args:
            start (float): The starting target normilized entropy 
            end (float): The ending target normilized entropy
            midpoint (float): Midpoint of the sigmoid decay of the target entropy
            max_steps(int): The number of steps that will be taken in training
            auto_tune (bool): If true will use start as the value for alpha
            alpha_value (float): The alpha value that will be used if not autotuning
        """
        
        self._start = start
        self._end = end
        self._midpoint = midpoint
        self._slope = slope
        self._log_alpha = torch.zeros(1, requires_grad=True)
        self._max_steps = max_steps
        self._auto_tune = auto_tune
        self._alpha_value = alpha_value
        
    def __call__(self) -> Tensor:
        """Will give the current alpha"""
        if not self._auto_tune:
            return self._alpha_value
        return self._log_alpha.exp()
    
    def sigmoid_target_entropy(self, current_step : int) -> float:
        """Will use sigmoid decay to calculate the target entropy

        Args:
            current_step (int): The current step in training

        Returns:
            float: The current target entropy
        """
        
        x = current_step / self._max_steps  # Normalize step to [0,1]
        return self._start + (self._end - self._start) / (1 + math.exp(-self._slope * (x - self._midpoint)))
    
    def to(self, device : torch.device) -> None:
        """Will move the alpha parameter to the device

        Args:
            device (torch.device): Torch device
        """
        
        self._log_alpha.to(device)
 
class Actor(nn.Module):
    
    def __init__(self,
                 embed_dim : int,
                 action_dim : int):
        super().__init__()
        self._device_select = nn.Linear(embed_dim, 1)
        self._action_select = nn.Linear(embed_dim, action_dim)
        self._action_dim = action_dim
        
    def forward(self, embed_states : Tensor, batch_index : Tensor) -> Tensor:
        """Will calculate the probs and log probs of taking a specific action on a device

            embed_states: a bmxe matrix where b is the batch size, m is the variable size of devices in each part group of the batch and e is the embeding dimension
            batch_index: bmx1, each element is the group that the element in the corresponding embed_state belongs to
            
            return: A bmxa tensor with logits for the possible actions inside of an element of the batch
        """
        
        device_select = self._device_select(embed_states) #bmx1
        action_select = self._action_select(embed_states) #bmxa
        b = batch_index.unique().numel()
        if b == 1:
            log_prob_dev = F.log_softmax(device_select, dim=-1)
        else:
            log_prob_dev = scatter_log_softmax(device_select.squeeze(), batch_index.squeeze()).view(-1,1)
        log_prob_act = F.log_softmax(action_select, dim=-1)
        
        return log_prob_act + log_prob_dev #Using rules of logs to have log(p_a * p_d) = log(p_a) + log(p_d)
    
    def sample_action(self, embed_states : Tensor, batch_index : Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Will sample an action for each group in the batch according to the given distribution
        
            embed_states: a bmxe matrix where b is the batch size, m is the variable size of devices in each part group of the batch and e is the embeding dimension
            batch_index: bmx1, each element is the group that the element in the corresponding embed_state belongs to
            
            return: bx1 tensor representing the index of the choosen action, b x (max m * a) of the padded probabilites, b x (max m * a) of the padded log probabilies
        
        """
        
        log_probs = self(embed_states, batch_index)
        log_probs_reshape = _reshape(log_probs, batch_index, self._action_dim)
        dist = Categorical(logits=log_probs_reshape)
        actions = dist.sample()
        probs = dist.probs
        return actions, probs, log_probs_reshape
        
    def evaluate_action(self, embed_states : Tensor, batch_index : Tensor) -> Tensor:
        """Will find and return the action that has the maximum probability of being choosen
        
            embed_states: a bmxe matrix where b is the batch size, m is the variable size of devices in each part group of the batch and e is the embeding dimension
            batch_index: bmx1, each element is the group that the element in the corresponding embed_state belongs to
            
            return: bx1 tensor representing the index of the choosen action, b x (max m * a) of the padded probabilites, b x (max m * a) of the padded log probabilies
        
        """
        log_probs = self(embed_states, batch_index)
        log_probs_reshape = _reshape(log_probs, batch_index, self._action_dim)
        return torch.argmax(log_probs_reshape, dim=-1)

        
   
class QModel(nn.Module):
    def __init__(self,
                embed_dim : int,
                action_dim : int):
        super().__init__()
        self._device_q = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 2))
        self._action_q = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, action_dim))
        self._action_dim = action_dim
        
    def forward(self, embed_state : Tensor, batch_index : Tensor, state_index : Tensor, description : str = None) -> Tensor:
        """Will calculate the Q value for each action on every device passed in

            Will first calculate the Q value of choosing the action on the device and not choosing an aciton on the device
            Will use then calculate the Q values for the action on the device.
            The q value for the actions on the device will be scaled by adding the q value of choosing the action on the device.
            The q values for the actions on the other devices will be scaled by the q value of choosing an action on another device
        
           embed_states: a bmxe matrix where b is the batch size, m is the variable size of devices in each part group of the batch and e is the embeding dimension
           batch_index: bmx1, each element is the group that the element in the corresponding embed_state belongs to
           state_index: each consecutive rolling pair of elements form a range of elements belonging to a group as follows [si[0], si[1]), [si[1], si[2]) .... [s[i], s[i+1]). Each is the set of indices of embed_states where each set corresponds to a group in the batch
           
           return: A bx(max_d * a) tensor where each element corresponds to the Q value of that specific device with zeroed out 
        """
        
        device_q : Tensor = self._device_q(embed_state) # bmx1 first index choose device second index choose other devicess
        action_q : Tensor = self._action_q(embed_state) # bmxa
        
        action_q += device_q[:,0].view(-1,1)
        
        scaler = torch.tensor([state_index[i + 1] - state_index[i] for i in range(len(state_index) - 1)])[batch_index]
        
        #Following essentially takes average of all other device q values. We sum every single one and then subtract our own from it and then divide the total amount of other devices. There is an edge case when there is no other devices.
        #TODO switch to pointer version of segment as segment_coo is non deterministic
        action_q += ((segment_coo(device_q[:,1], batch_index, reduce='sum')[batch_index]- device_q[:, 1]) / (scaler - 1 + EPS)).view(-1,1) #Need EPS so that I am not dividing by zero when there is no other device. This will still end up being zero since the left hand side will be zero
        
        if description:
            LearningLogger().log_scalars({f'{description} Mean' : action_q.mean(), 
                                      f'{description} Max' : action_q.max(),
                                      f'{description} Min' : action_q.min(),
                                      f'{description} Std' : action_q.std()}, steps=LearningLogger().cur_step())
        
        return _reshape(action_q, batch_index, self._action_dim, filler_val=0)

class QFunction(nn.Module):
    
    def __init__(self, 
                 embed_dim : int,
                 action_dim : int):
        super().__init__()
        self._q1 = QModel(embed_dim, action_dim)
        self._q2 = QModel(embed_dim, action_dim)
        
    def forward(self, embed_state : Tensor, batch_index : Tensor, state_index : Tensor) -> tuple[Tensor, Tensor]:
        q1 = self._q1(embed_state, batch_index, state_index, description='Q1')
        q2 = self._q2(embed_state, batch_index, state_index, description='Q2')
        
        return q1, q2
    
class QFunctionTarget():
    
    def __init__(self, qfunction : QFunction, tau : float):
        self._target_q_function = deepcopy(qfunction)
        self._actual_q_function = qfunction
        self._tau = tau
        
    def __call__(self, *args, **kwds):
        return torch.min(torch.stack(self._actual_q_function(*args, **kwds), dim=2), dim=2)[0]
    
    def update(self):
        """Will do polyak averaging to each model in the target"""
        for param, target_param in zip(self._actual_q_function._q1.parameters(), self._target_q_function._q1.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        for param, target_param in zip(self._actual_q_function._q2.parameters(), self._target_q_function._q2.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
    
    def to(self, device : torch.device):
        self._target_q_function.to(device)

def _reshape(logits : Tensor, batch_index : Tensor, action_dim : int, filler_val : float = -1e8) -> Tensor:
    """Will reshape the logits from the form bmxa to bxma where m is the variable about of devices. Since it is variable, the rows will be padded with zeros when necessary
    
    embed_states: a bmxe matrix where b is the batch size, m is the variable size of devices in each part group of the batch and e is the embeding dimension
    batch_index: bmx1, each element is the group that the element in the corresponding embed_state belongs to
        
    return: bxma matrix
    
    """
    a = action_dim
    b = batch_index.unique().numel()
    max_d = torch.bincount(batch_index).max().item()

    # Step 2: Compute positions within each group
    device_counts = torch.zeros(b, dtype=torch.long).scatter_add_(
        0, batch_index, torch.ones_like(batch_index)
    )
    device_offsets = torch.cumsum(
        torch.cat([torch.tensor([0], dtype=torch.long), device_counts[:-1]]), dim=0
    )

    device_indices = (torch.arange(len(batch_index)) - device_offsets[batch_index]).long()

    # Step 3: Prepare an output tensor with zeros
    output = torch.zeros((b, max_d * a), dtype=logits.dtype) + filler_val
    # Step 4: Place actions into the reshaped matrix
    row_indices = batch_index
    col_indices = (device_indices[:, None] * a + torch.arange(a)).flatten()

    return output.index_put_((row_indices.repeat_interleave(a), col_indices), logits.flatten())