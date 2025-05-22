import math

import torch.nn as nn
import torch
import torch_geometric
import torch_geometric.nn as GeoNN
from torch.nn.functional import leaky_relu
from torch_geometric.nn import MessagePassing, GlobalAttention, TransformerConv, GATv2Conv, LayerNorm, global_mean_pool, global_max_pool


#Code from https://github.com/jaromiru/NASimEmu-agents/blob/main/graph_nns.py#L363

class MultiMessagePassing(nn.Module):
    def __init__(self, steps, emb_dim):
        super().__init__()

        self._gnns = nn.ModuleList( [GraphNet(emb_dim) for i in range(steps)] )           
        self._emb_dim = emb_dim
        self._steps = steps

    def forward(self, x, edge_attr, edge_index, batch_ind, num_graphs, data_lens):
        x_global = torch.zeros(num_graphs, self._emb_dim)

        for i in range(self._steps):
            x = self._gnns[i](x, edge_attr, edge_index, x_global, batch_ind)            

        return x

class MultiMessagePassingWithAttention(nn.Module):
    def __init__(self, steps, emb_dim):
        super().__init__()

        self.gnns = nn.ModuleList( [GraphNet(emb_dim) for _ in range(steps)] )
        self.att  = nn.ModuleList( [GATv2Conv(emb_dim, emb_dim, add_self_loops=False, heads=3, concat=False) for i in range(steps - 1)] )
        self._emb_dim = emb_dim
        self._steps = steps

    def forward(self, x, edge_attr, edge_index, batch_ind, num_graphs, data_lens):
        x_att = torch.zeros(len(x), self._emb_dim)

        edge_complete = complete_graph(data_lens)

        for i in range(self._steps):
            x = self.gnns[i](x, edge_attr, edge_index, x_att, batch_ind)
            if i < self._steps - 1:
                x_att = leaky_relu( self.att[i](x, edge_complete) )

        return x


class MultiMessagePassingWithTransformer(nn.Module):
    def __init__(self, steps, emb_dim):
        super().__init__()

        self._gnns = nn.ModuleList( [GraphNet(emb_dim) for _ in range(steps)] )
        self._att  = nn.ModuleList( [GeoNN.TransformerConv(emb_dim, emb_dim, heads=3, concat=False) for i in range(steps - 1)] )
        self._emb_dim = emb_dim

        self._steps = steps

    def forward(self, x, edge_attr, edge_index, batch_ind, data_lens):
        x_att = torch.zeros(len(x), self._emb_dim)
        edge_complete = complete_graph(data_lens)

        for i in range(self._steps):
            x = self._gnns[i](x, edge_attr, edge_index, x_att, batch_ind)
            if i < self._steps - 1:
                x_att = leaky_relu( self._att[i](x, edge_complete) )

        return x

class GraphNet(MessagePassing):
    def __init__(self, emb_dim : int):
        super().__init__(aggr='max')

        # self.f_mess = Sequential( Linear(config.emb_dim + config.edge_dim, config.emb_dim), LeakyReLU(),
        #     Linear(config.emb_dim, config.emb_dim), LeakyReLU(),
        #     Linear(config.emb_dim, config.emb_dim), LeakyReLU())

        # self.f_agg  = Sequential( Linear(config.emb_dim + config.emb_dim + config.emb_dim, config.emb_dim), LeakyReLU(),
        #     Linear(config.emb_dim, config.emb_dim), LeakyReLU(),
        #     Linear(config.emb_dim, config.emb_dim), LeakyReLU())

        self.f_mess = nn.Sequential( nn.Linear(emb_dim, emb_dim), nn.LeakyReLU() ) #Edge dim in original code is zero in the config file
        self.f_agg  = nn.Sequential( nn.Linear(emb_dim + emb_dim + emb_dim, emb_dim), nn.LeakyReLU() )

    def forward(self, x, edge_attr, edge_index, xg, batch_ind):
        xg = xg[batch_ind] # expand
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, xg=xg)

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            z = torch.cat([x_j, edge_attr], dim=1)
        else:
            z = x_j

        z = self.f_mess(z)

        return z 

    def update(self, aggr_out, x, xg):
        z = torch.cat([x, xg, aggr_out], dim=1)
        z = self.f_agg(z) + x # skip connection

        return z  
    
def complete_matrix(data_lens):
    size = sum(data_lens)
    complete_adj = torch.zeros(size, size)
    
    start = 0
    for l in data_lens:
        complete_adj[start:start+l,start:start+l] = 1
        start += l

    # complete_adj -= torch.eye(size)  # remove self-connections
    return complete_adj.unsqueeze(0)

def complete_graph(data_lens):
    complete_adj = complete_matrix(data_lens)
    edge_index_complete, _ = torch_geometric.utils.dense_to_sparse(complete_adj)

    return edge_index_complete


