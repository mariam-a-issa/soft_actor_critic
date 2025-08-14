
'''

Graph Bundle Bind Encoder:

Bundle each nodes features together to make a node hypervector then normalize.
Bundle the node hypervectors of the same subnet together then normalize.
Do a symmetric bind of each connected subnet and normalize (by sqrt 2 or norm). (s_1 * p(s_2) + s_2 * p(s_1))
Bundle all of the subnet connections together and normalize.
Permute the resulting graph by one then bind the original node.

Note: if we were to do bipolar hypervectors then we could normalize by the sqrt of the number of whatever we bundle
      if we were to do random gaussian hypervectors then we would normalize by the norm

'''
import math

import torch
import torch.nn.functional as F
from torch import tensor, Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_add

from ...model_utils import permute_rows_by_shifts, generate_counting_tensor

class GBUBIEncoder:

    def __init__(self, 
                 dim : int,
                 node_dim : int,
                 bipolar : bool = False,
                 variance : float = 1) -> None:
        """Will create Graph Bundle Bind Encoder 

        Args:
            dim (int): The dimension of the hypervector
            node_dim (int): The dimension of each node that would be in the state
            bipolar (bool): Determine whether or not to use bipolar hypervectors
        """
        
        self._bipolar = bipolar
        self._dim = dim
        self._node_dim = node_dim
        
        self._subnet_base = torch.randn(node_dim-1, dim) * math.sqrt(variance)
        self._node_base = torch.randn(node_dim, dim) * math.sqrt(variance)
        self._discovery_base = torch.randn(1, dim) * math.sqrt(variance)
        self._bias = 2 * math.pi * torch.randn(1, dim)
        

    def __call__(self, nodes : Batch, state_index : Tensor) -> tuple[Tensor, Tensor]:
        """Will encode each node into a hyperdimensional representation

        Args:
            nodes (Batch): The nodes that will be encoded. Will be an mxd matrix where there are a total of m nodes each with an embedding dim of d
            state_index (Tensor): An array Where each rolling pair of elements represents the range of devices in a certain batch 

        Returns:
            tuple[Tensor, Tensor]: First element is the encoded state, the second element is the batch index to save computation
        """

        #Setup Tensors
        is_subnet = nodes.x[:, 0] == 1
        node_features = 2 * nodes.x[~is_subnet].float() - 1 # n x f. Number of nodes x number of features. Do 2*x - 1 for hamming distance
        encoded_features = torch.exp(1j * (node_features[:, 1:] @ self._subnet_base + self._bias[0])) # n x d

        #Build Subnets
        adj_matrix = to_dense_adj(nodes.edge_index).squeeze()
        subnet_node_adj_matrix = adj_matrix[is_subnet][:, ~is_subnet] # sub_nets x n
        encoded_subnet = subnet_node_adj_matrix.to(torch.cfloat) @ encoded_features #sub_nets x d
        perm_encoded_subnet = permute_rows_by_shifts(encoded_subnet, torch.ones(encoded_subnet.shape[0], dtype=torch.int))
        
        #Build Subnet Connections
        binded_subnets = self._bind_subnets_hadamard(encoded_subnet, adj_matrix[is_subnet][:, is_subnet], perm_encoded_subnet) #sub_nets x d
        
        #Build Graph
        subnet_batch_idx = nodes.batch[is_subnet]
        num_graphs = int(subnet_batch_idx.max()) + 1
        bundled = scatter_add(binded_subnets, subnet_batch_idx, dim=0, dim_size=num_graphs)  # graph x d
        expanded_graphs = bundled[nodes.batch[~is_subnet]]

        # Add discovery feature (may need to try sin cosine encoding)
        device_batch = nodes.batch[~is_subnet]
        device_count = torch.bincount(device_batch, minlength=int(device_batch.max().item()) + 1)[device_batch]
        device_index = generate_counting_tensor(state_index)
        prop_devices : Tensor = (device_index + 1) / device_count
        node_features = torch.cat([node_features[:, 1:]], prop_devices.view(-1, 1), dim=1)
        
        #Encode Device
        encoded_nodes = torch.exp(1j * (node_features @ self._node_base)) # n x d #Do not need a bias as we already have it from the graphs earlier
        encoded_nodes = encoded_nodes * expanded_graphs

        return encoded_nodes, nodes.batch[~is_subnet]

    def _bind_subnets_hadamard(self, 
        encoded_subnet: torch.Tensor,        # [N, D]
        subnet_adj: torch.Tensor,            # [N, N] (symmetric; 0/1 or weights)
        perm_encoded_subnet: torch.Tensor,   # [N, D] (pre-permuted subnet embeddings)
        degree_norm: bool = False
    ) -> torch.Tensor:
        """
        out[i] = sum_j A[i,j] * ( encoded_subnet[i] ⊙ perm_encoded_subnet[j] )
        Returns: [N, D]
        """
        E = encoded_subnet           # [N, D]
        P = perm_encoded_subnet      # [N, D]
        A = subnet_adj               # [N, N]

        A.fill_diagonal_(1) #Needed for subnet

        N, D = E.shape
        # Broadcast to [N, N, D]: rows i receive from columns j
        Ei = E[:, None, :].expand(N, N, D)     # receivers
        Pj = P[None, :, :].expand(N, N, D)     # neighbors (already permuted)

        # Hadamard bind, mask by adjacency, sum over neighbors
        M = Ei * Pj                             # [N, N, D]
        out = (A.unsqueeze(-1) * M).sum(dim=1)  # [N, D]

        if degree_norm:
            deg = A.sum(dim=1, keepdim=True).clamp_min(1).to(out.dtype)
            out = out / deg

        return out


    def to(self, device : torch.device) -> None:
        self._base.to(device)
        self._subnet_base.to(device)
        self._bias.to(device)
