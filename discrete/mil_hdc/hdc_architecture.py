import torch
from torch import tensor, Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_log_softmax, segment_coo
from torch.distributions import Categorical
from copy import deepcopy

from utils import LearningLogger, EPS
from ..mil_utils import generate_batch_index, generate_counting_tensor, permute_rows_by_shifts, permute_rows_by_shifts_matrix, reshape

#TODO Switch it up so that it does not bind them down at the end and only binds them to gether then permutes
class Encoder:
    
    def __init__(self,
                 dim : int, 
                 node_dim : int) -> None:
        """Will create an HDC encoder

        Args:
            dim (int): The dimension of the hypervector
            node_dim (int): The dimension of each node that would be in the state
            distribution (str): The distribution used to build the RHFF basis vectors
        """
        
        self._s_hdvec = torch.randn(node_dim, dim, dtype=torch.float32) 
        self._bias = 2 * math.pi * torch.rand(dim, dtype=torch.float32)
        self._dim = dim
        
        
    def __call__(self, nodes : Tensor, state_index : Tensor) -> tuple[Tensor, Tensor]:
        """Will encode each node into a hyperdimensional representation

        Args:
            nodes (Tensor): The nodes that will be encoded. Will be an mxd matrix where there are a total of m nodes each with an embedding dim of d
            state_index (Tensor): An array Where each rolling pair of elements represents the range of devices in a certain batch 

        Returns:
            tuple[Tensor, Tensor]: First element is the encoded state, the second element is the batch index to save computation
        """
        
        #m x hyper_dim = m x feat @ feat x hyper_dim
        
        # encoded_devices = torch.exp(1j * ((nodes @ self._s_hdvec) + self._bias))
        # permute_matrix = generate_counting_tensor(state_index)
        # devices_permuted = permute_rows_by_shifts(encoded_devices, permute_matrix)
        # batch_index = generate_batch_index(state_index)
        # #Will bind together all of the devices in a single batch
        # write = torch.ones(len(torch.unique(batch_index)), self._dim, dtype = torch.cfloat)
        # write.scatter_reduce_(dim=0, src=devices_permuted, index=batch_index.view(-1, 1).expand(-1 , self._dim), reduce='prod')
        
        #Encode and permute the devices
        encoded_devices = nodes @ self._s_hdvec + self._bias
        permute_vector = generate_counting_tensor(state_index)
        devices_permuted = permute_rows_by_shifts(encoded_devices, permute_vector)
        
        #Bind them all together by adding them then exp
        batch_index = generate_batch_index(state_index)
        grouped_products : Tensor = torch.zeros((batch_index.max() + 1, encoded_devices.shape[1]), dtype=torch.cfloat)
        devices_permuted = torch.exp(1j * devices_permuted)
        grouped_products.index_add_(0, batch_index, devices_permuted)
        grouped_products = grouped_products[batch_index] 

        #Repermute them so that the specific device aligns
        final_encode = permute_rows_by_shifts(grouped_products, -1 * permute_vector)
        
        return final_encode, batch_index
    
    def to(self, device : torch.device) -> None:
        self._s_hdvec.to(device)
        self._bias.to(device)
    
    
class Actor(nn.Module):
    
    def __init__(self, 
                 dim : int,
                 action_dim : int) -> None:
        """Init an HDC based actor

        Args:
            dim (int): The dimension of the hypervector
            action_dim (int): The number of actions that can be taken at a node
        """
        super().__init__()
        self._action = nn.Linear(dim, action_dim, dtype=torch.cfloat, bias=False)
        self._device = nn.Linear(dim, 1, dtype = torch.cfloat, bias=False)

        self._action.weight = nn.Parameter(torch.zeros(action_dim, dim, dtype=torch.cfloat))
        self._device.weight = nn.Parameter(torch.zeros(1, dim, dtype=torch.cfloat))
        
        self._dim = dim
        self._action_dim = action_dim
        
    def forward(self, embedded_state : Tensor, batch_index : Tensor, state_index : Tensor) -> Tensor:
        """Will return the values after

        Args:
            encoded_state (Tensor): A b x d matrix where there are b encoded devices with dimension d. Note each row represents all the devices at a single step. There can be a variable amount of devices at each step
            batch_index (Tensor): An array where each element represents what element of the batch that the corresponding row belongs to
            state_index (Tensor): An array representing the ranges of devices in a batch

        Returns:
            Tensor: bm x a matrix where it represents the logits of actions on a device
        """
        # counting = generate_counting_tensor(state_index)
        # BM = counting.shape[0] #Bm is the total number of devices
        # permuted_a_model : Tensor = permute_rows_by_shifts_matrix(self._action.unsqueeze(0).expand(BM, self._action_dim, self._dim),
        #                                                          counting)
        # permuted_d_model : Tensor = permute_rows_by_shifts_matrix(self._device.unsqueeze(0).expand(BM, 1 , self._dim),
        #                                                          counting)
        
        # #go from bmxaxd @ bxd-> bmxaxd @ bxdx1 -> bmxaxd @ bmxdx1 = b x a x 1
        # device_select = torch.real((permuted_d_model @ embedded_state[batch_index].unsqueeze(-1)).view(BM, 1)) / self._dim
        # action_select = torch.real((permuted_a_model @ embedded_state[batch_index].unsqueeze(-1)).view(BM, self._action_dim)) / self._dim
        #TODO may need to not normalize
        action_select = torch.real(self._action(torch.conj(embedded_state))) / self._dim
        device_select = torch.real(self._device(torch.conj(embedded_state))) / self._dim
        
        b = batch_index.unique().numel()
        if b == 1:
            log_prob_dev = F.log_softmax(device_select, dim=-1)
        else:
            log_prob_dev = scatter_log_softmax(device_select.squeeze(), batch_index.squeeze()).view(-1,1)
        log_prob_act = F.log_softmax(action_select, dim=-1)
        
        return log_prob_act + log_prob_dev #Using rules of logs to have log(p_a * p_d) = log(p_a) + log(p_d)
    
    def sample_action(self, embed_states : Tensor, batch_index : Tensor, state_index : Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Will sample an action for each group in the batch according to the given distribution

            Args:
                embed_states (Tensor): a bmxe matrix where b is the batch size, m is the variable size of devices in each part group of the batch and e is the embeding dimension
                batch_index (Tensor): bmx1, each element is the group that the element in the corresponding embed_state belongs to
                state_index (Tensor): An array representing the ranges of devices in a batch
            
            Returns:
                Tensor: bx1 tensor representing the index of the choosen action, b x (max m * a) of the padded probabilites, b x (max m * a) of the padded log probabilies
        
        """
        
        log_probs = self(embed_states, batch_index, state_index)
        log_probs_reshape = reshape(log_probs, batch_index, self._action_dim)
        dist = Categorical(logits=log_probs_reshape)
        actions = dist.sample()
        probs = dist.probs
        return actions, probs, log_probs_reshape
    
    def evaluate_action(self, embed_states : Tensor, batch_index : Tensor, state_index : Tensor) -> Tensor:
        """Will find and return the action that has the maximum probability of being choosen
        
            embed_states: a bmxe matrix where b is the batch size, m is the variable size of devices in each part group of the batch and e is the embeding dimension
            batch_index: bmx1, each element is the group that the element in the corresponding embed_state belongs to
            
            return: bx1 tensor representing the index of the choosen action, b x (max m * a) of the padded probabilites, b x (max m * a) of the padded log probabilies
        
        """
        log_probs = self(embed_states, batch_index, state_index)
        log_probs_reshape = reshape(log_probs, batch_index, self._action_dim)
        return torch.argmax(log_probs_reshape, dim=-1)
    
    
class QModel():
    
    def __init__(self, 
                 dim : int,
                 action_dim : int) -> None:
        """Init a HDC based Critic

        Args:
            dim (int): Dim of the hypervector
            action_dim (int): Number of actions that can be taken
        """
        upper_bound = 1 / math.sqrt(dim)
        lower_bound = -upper_bound
        
        #Using the same initilzation as the torch.nn.Linear 
        #https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L106-L108

        self._action = (upper_bound - lower_bound) * torch.rand(dim, action_dim, dtype=torch.cfloat) + lower_bound
        self._action.requires_grad_(False)
        self._device = torch.zeros(dim, 2, dtype=torch.cfloat, requires_grad=False)
        
        self._dim = dim
        self._action_dim = action_dim
        
    def _values(self, embedded_state : Tensor, batch_index : Tensor, state_index : Tensor) -> tuple[Tensor, Tensor]:
        """Will get the Q values for the specific actions and then for the device

        Args:
            embedded_state (Tensor): The embedding states with a dim of bxd. Each element contains all of the devices at that step binded together
            batch_index (Tensor): An array where each element represents what element of the batch that the corresponding row belongs to
            state_index (Tensor): An array representing the ranges of devices in a batch

        Returns:
            tuple[Tensor, Tensor]: Will return the Q values for the action (bm x a) and then for the devices (bm x 2)
        """
        
        # counting = generate_counting_tensor(state_index)
        # BM = counting.shape[0] #Bm is the total number of devices
        # permuted_a_model : Tensor = permute_rows_by_shifts_matrix(self._action.unsqueeze(0).expand(BM, self._action_dim, self._dim),
        #                                                          counting)
        # permuted_d_model : Tensor = permute_rows_by_shifts_matrix(self._device.unsqueeze(0).expand(BM, 2 , self._dim),
        #                                                          counting)
        

        action_q = torch.real(torch.conj(embedded_state) @ self._action) / self._dim
        device_q = torch.real(torch.conj(embedded_state) @ self._device) /self._dim

        num_devices = torch.diff(state_index)
        action_q /= num_devices.unsqueeze(dim=-1) #Need to normalize q value as we are using bundeling for the encoding
        
        return action_q, device_q
    
    def __call__(self, embedd_state : Tensor, batch_index : Tensor, state_index : Tensor, description : str = None) -> Tensor:
        """Will give the q values for each possible action at a certain state

        Args:
            embedd_state (Tensor): The embedding states with a dim of bxd. Each element contains all of the devices at that step binded together
            batch_index (Tensor): An array where each element represents what element of the batch that the corresponding row belongs to
            state_index (Tensor): An array representing the ranges of devices in a batch
            description (str): Information about this current Q Model to help with logging

        Returns:
            Tensor: A bm x a array with the Q values
        """
        
        action_q : Tensor
        device_q : Tensor
        action_q, device_q = self._values(embedd_state, batch_index, state_index)
        
        #action_q += device_q[:,0].view(-1,1)
        
        #scaler = torch.tensor([state_index[i + 1] - state_index[i] for i in range(len(state_index) - 1)])[batch_index]
        
        #Following essentially takes average of all other device q values. We sum every single one and then subtract our own from it and then divide the total amount of other devices. There is an edge case when there is no other devices.
        #TODO switch to pointer version of segment as segment_coo is non deterministic
        #action_q += ((segment_coo(device_q[:,1], batch_index, reduce='sum')[batch_index]- device_q[:, 1]) / (scaler - 1 + EPS)).view(-1,1) #Need EPS so that I am not dividing by zero when there is no other device. This will still end up being zero since the left hand side will be zero
        
        if description:
            LearningLogger().log_scalars({f'{description} Mean' : action_q.mean(), 
                                      f'{description} Max' : action_q.max(),
                                      f'{description} Min' : action_q.min(),
                                      f'{description} Std' : action_q.std()}, steps=LearningLogger().cur_step())
            
        return reshape(action_q, batch_index, self._action_dim, filler_val=0)
    
    def parameters(self) -> Tensor:
        """Will return the model hypervector 

        Returns:
            Tensor: Actions model
        """
        return self._action.T
    
    def to(self, device : torch.device)-> None:
        self._action.to(device)
        self._device.to(device)

class QFunction():
    
    def __init__(self, 
                 embed_dim : int,
                 action_dim : int):
        self._q1 = QModel(embed_dim, action_dim)
        self._q2 = QModel(embed_dim, action_dim)
        
    def __call__(self, embed_state : Tensor, batch_index : Tensor, state_index : Tensor) -> tuple[Tensor, Tensor]:
        q1 = self._q1(embed_state, batch_index, state_index, description='Q1')
        q2 = self._q2(embed_state, batch_index, state_index, description='Q2')
        
        return q1, q2
    
    def parameters(self) -> tuple[Tensor, Tensor]:
        """Will return the parameters of both of the q models

        Returns:
            tuple[Tensor, Tensor]: The parameters of both of the q models
        """
    
        return self._q1.parameters(), self._q2.parameters()
    
    def to(self, device : torch.device)->None:
        self._q1.to(device)
        self._q2.to(device)

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
