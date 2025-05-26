import math

import torch
from torch import Tensor
from ...model_utils import permute_rows_by_shifts, generate_batch_index, generate_counting_tensor, positional_encoding

class Encoder:
    
    def __init__(self,
                 dim : int, 
                 node_dim : int,
                 pos_enc_dim : int) -> None:
        """Will create an HDC encoder

        Args:
            dim (int): The dimension of the hypervector
            node_dim (int): The dimension of each node that would be in the state
            distribution (str): The distribution used to build the RHFF basis vectors
        """
        
        self._feat_s_hdvec = torch.randn(2, node_dim, dim, dtype=torch.float32) 
        #self._feat_bias = 2 * math.pi * torch.rand(dim, dtype=torch.float32)
        self._pos_s_hdvcec = torch.randn(pos_enc_dim, dim, dtype=torch.float32)
        self._pos_bias = 2 * math.pi * torch.rand(dim, dtype=torch.float32)
        self._dim = dim
        self._pos_enc_dim = pos_enc_dim
        
        
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
        
        #Generate helper vectors
        index_vector = generate_counting_tensor(state_index)
        batch_index = generate_batch_index(state_index)
        number_nodes = torch.diff(state_index)[batch_index].view(-1, 1)
        
        #Encode the nodes
        pos_enc = positional_encoding(index_vector, self._pos_enc_dim)
        #First get correct matrix for each node based on binary value, then pick the right vector for each node, then exponent, then bundle, then normalize
        hd_feats = torch.exp(1j * self._feat_s_hdvec[nodes.to(torch.int64).clamp(max=1), torch.arange(nodes.shape[1])]).sum(dim=1) / nodes.shape[1]
        hd_pos = torch.exp(1j * (pos_enc @ self._pos_s_hdvcec + self._pos_bias))
        encoded_nodes = hd_feats * hd_pos
        
        # #Bundle them total state nodes together and normalize
        grouped_products : Tensor = torch.zeros((batch_index.max() + 1, encoded_nodes.shape[1]), dtype=encoded_nodes.dtype)
        grouped_products.index_add_(0, batch_index, encoded_nodes)
        grouped_products = grouped_products[batch_index]
        grouped_products /= number_nodes
        
        #Permute total state by one
        grouped_products = permute_rows_by_shifts(grouped_products, torch.ones(grouped_products.shape[0], dtype=torch.int64))
        
        #Bind total state and each node and normalize
        encoded_nodes = encoded_nodes * grouped_products
        
        return encoded_nodes, batch_index
    
    def to(self, device : torch.device) -> None:
        self._feat_s_hdvec.to(device)
        self._pos_s_hdvcec.to(device)
        self._pos_bias.to(device)