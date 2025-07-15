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
        
        self._s_hdvec = torch.randn(node_dim + pos_enc_dim, dim, dtype=torch.float32) 
        self._bias = 2 * math.pi * torch.rand(dim, dtype=torch.float32)
        self._node_dim = node_dim
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
        
        index_vector = generate_counting_tensor(state_index)
        batch_index = generate_batch_index(state_index)
        number_nodes = torch.diff(state_index)[batch_index].view(-1, 1)

        #Encode the nodes
        pos_enc = positional_encoding(index_vector, self._pos_enc_dim)
        nodes = torch.cat((nodes, pos_enc), dim = 1)
        encoded_nodes = nodes @ self._s_hdvec + self._bias
        encoded_nodes = torch.exp(1j * encoded_nodes)
        
        #Bundle them total state nodes together
        grouped_products : Tensor = torch.zeros((batch_index.max() + 1, encoded_nodes.shape[1]), dtype=encoded_nodes.dtype)
        grouped_products.index_add_(0, batch_index, encoded_nodes)
        grouped_products = grouped_products[batch_index]
        
        #Permute total state by one
        grouped_products = permute_rows_by_shifts(grouped_products, torch.ones(grouped_products.shape[0], dtype=torch.int64))
        
        #Bind total state and each node and normalize
        final_encode = encoded_nodes + grouped_products / number_nodes
        return final_encode, batch_index
        
    def to(self, device : torch.device) -> None:
        self._s_hdvec.to(device)
        self._bias.to(device)