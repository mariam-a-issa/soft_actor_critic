
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

import torch
import torch.nn.functional as F
from torch import tensor

class GBUBIEncoder:

    def __init__(self, 
                 dim : int,
                 node_dim : int,
                 bipolar : bool) -> None:
        """Will create Graph Bundle Bind Encoder 

        Args:
            dim (int): The dimension of the hypervector
            node_dim (int): The dimension of each node that would be in the state
            bipolar (bool): Determine whether or not to use bipolar hypervectors
        """
        
        self._bipolar = bipolar
        self._dim = dim
        self._node_dim = node_dim
        
        self._base = torch.randn(node_dim, dim)

        if bipolar:
            self._base = torch.where(self._base < 0, torch.tensor(-1.0), torch.tensor(1.0))
        else:
            self._base = F.normalize(self._base, p=2, dim=1)

        

    def __call__(self, nodes : Batch, state_index : Tensor) -> tuple[Tensor, Tensor]:
        """Will encode each node into a hyperdimensional representation

        Args:
            nodes (Tensor): The nodes that will be encoded. Will be an mxd matrix where there are a total of m nodes each with an embedding dim of d
            state_index (Tensor): An array Where each rolling pair of elements represents the range of devices in a certain batch 

        Returns:
            tuple[Tensor, Tensor]: First element is the encoded state, the second element is the batch index to save computation
        """

        is_subnet = nodes.x[:, 0] == 1
        node_features = nodes.x[~is_subnet] # n x f. Number of nodes x number of features

        #Basically build the node hypervectors
        #Build subnet hypervectors by cutting down adjancy matrix to (num_subnets, num_devices)
        #Create symmetric binds

        #I hope to not use the scatter operations and start to do padding instead. Scatter is seeming to be more slow.