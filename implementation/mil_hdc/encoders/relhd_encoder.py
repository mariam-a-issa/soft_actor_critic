import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_sparse import SparseTensor


class RelHD(object):
    def __init__(self, dim : int,
                       node_dim : int,
                       bipolar = True, 
                       scale = 1):
        self._dimension = dim
        self._node_dim = node_dim
        self._bipolar = bipolar
        self._scale = scale
        
        self._base_hvecs = torch.randn(node_dim, dim) # f x d
        self._base_bipolar = torch.where(self._base_hvecs < 0, torch.tensor(-1.0), torch.tensor(1.0))
        
        #embedding features for node hypervector, 1hop, 2hop
        self._phi0 = torch.where(torch.randn(self._dimension) < 0, torch.tensor(-1.0), torch.tensor(1.0))
        self._phi1 = torch.where(torch.randn(self._dimension) < 0, torch.tensor(-1.0), torch.tensor(1.0))
        self._phi2 = torch.where(torch.randn(self._dimension) < 0, torch.tensor(-1.0), torch.tensor(1.0))
        

    #encodes the feature nodes into hypervectors
    def __call__(self, nodes : Batch):
        is_subnet = nodes.x[:, 0] == 1
        node_features = nodes.x[~is_subnet] # n x f. Number of nodes x number of features
        
        #simple feature matrix sampled from a gaussian distribution and matmul to get encoded node hypervectors
        if self._bipolar:
            encoded_nodes = node_features @ self._base_bipolar
        else:
            encoded_nodes = node_features @ self._base_hvecs
        
        row, col = nodes.edge_index
        N_tot = nodes.num_nodes
        
        #Need to do 2 and 4 hop because there will be subnet(s) between nodes
        #2 hop represents node in the same subnet. 4 hop represents node in neighboring subnet
        
        #Create Sparse matrix to find the 2/4-hops
        A = SparseTensor(row=row, col=col, sparse_sizes=(N_tot, N_tot))

        #Find the 2 hop
        A_2 = A @ A

        #Remove self loops and normalize 
        A_2_M = A_2.remove_diag().set_value_(lambda v: v>0, layout='coo')

        #Find the 4 hop
        A_4 = A_2_M @ A_2_M

        #Remove self loops. Normalze
        A_4_M = A_4.remove_diag().set_value_(lambda v: v>0, layout='coo')

        #Make dense
        A_2_M = A_2_M.to_dense()
        A_4_M = A_4_M.to_dense()

        #Remove A_2 repeats (nodes in same subnet not other subnet)
        A_4_M[A_2_M > 0] = 0

        #Remove subnet nodes
        keep = torch.ones(N_tot, dtype=torch.bool)
        keep[is_subnet] = False
        A_2_M = A_2_M[keep][:,keep]
        A_4_M = A_4_M[keep][:,keep]
        
        hop_2 = A_2_M @ encoded_nodes
        hop_4 = A_4_M @ encoded_nodes
        
        return encoded_nodes * self._phi0 + hop_2 * self._phi1 + hop_4 * self._phi2