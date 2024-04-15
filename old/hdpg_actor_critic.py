#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.utils.tensorboard import SummaryWriter
import random
from torch.distributions import MultivariateNormal
from copy import deepcopy

import mujoco_py

# Open GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# This is used for the Actor encoding
class RBFEncoder(nn.Module):
    def __init__(self, in_size: int, out_size: int = 4096):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        # TODO: may need change
        s_hdvec = torch.randn(self.in_size, self.out_size) / in_size
        bias = 2 * math.pi * torch.randn(self.out_size)
        # Encoder paprameter
        self.basis = nn.parameter.Parameter(s_hdvec)
        self.base = nn.parameter.Parameter(bias)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: need to check dimension of x
        x = x @ self.basis + self.base
        x = torch.cos(x)
        return x

# class HD_DDPG_encode(nn.Module):
#     def __init__(self, in_size, out_size=4096, dim_size=8192, lr=0.02, action_std_init=0.6):
#         super().__init__()
#         self.in_size = in_size
#         self.out_size = out_size  # TODO: out_size isn't use
#         self.dim_size = dim_size
#         self.encoder = RBFEncoder(in_size, dim_size)  # TODO: actual encoder
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x


class Actor(nn.Module):
    def __init__(self, in_size: int, out_size: int, dim_size=8192):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(dim_size, out_size, bias=False)

    def forward(self, s):
        a = self.l1(s)
        return a


class HD_Actor:
    def __init__(self, in_size: int, out_size: int, dim_size=8192, lr=0.02, action_std_init=0.6):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dim_size = dim_size
        self.model = Actor(in_size, out_size, dim_size)  # Actor is single layer D*A
        self.model.l1.weight.data = torch.zeros((out_size, dim_size))  # TODO: need to check sequence
        self.lr = lr
        self.softmax = nn.Softmax()
        self.action_var = torch.full((out_size,), action_std_init * action_std_init)

        self.encoded = None

    def set_action_std(self, new_std):
        self.action_var = torch.full((self.out_size,), new_std * new_std)

    def forward(self):
        x = self.model(self.encoded)

    def _pred(self, x):
        return self.model(x)

    def predict(self, x):
        # action_mean = self._pred(x)
        action_mean = self.model(x)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # TODO: why increase dimension?
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob


class EXPEncoder(nn.Module):
    def __init__(self, n_state, dimension):
        super(EXPEncoder, self).__init__()
        s_hdvec = []
        for n in range(n_state):
            s_hdvec.append(np.random.normal(0, 1, dimension))
        bias = np.random.uniform(0, 2 * np.pi, size=dimension)

        self.s_hdvec = nn.parameter.Parameter(torch.FloatTensor(s_hdvec))
        self.bias = nn.parameter.Parameter(torch.FloatTensor(bias))

        self.requires_grad_(False)

    def forward(self, state):
        return torch.exp(1j * (torch.matmul(state, self.s_hdvec) + self.bias))


class Critic(nn.Module):
    def __init__(self, dimension, n_action):
        super(Critic, self).__init__()
        self.D = dimension
        self.action = n_action
        self.l1 = nn.Linear(dimension, n_action, dtype=torch.cfloat)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        action = action.type(torch.cfloat)
        action = action.squeeze()
        state = state.type(torch.cfloat)
        temp = self.l1(state)
        # force data type change
        temp = temp.type(torch.cfloat)

        res = torch.matmul(temp, action)

        return res


class HD_Critic:
    def __init__(self, dimension, n_state, n_action, lr):
        self.D = dimension
        self.n_action = n_action
        self.n_state = n_state
        self.lr = lr

        self.model = Critic(dimension, n_action)  # TODO: DDPG critic model

        self.encoded = None

    def value(self, action: torch.Tensor):
        output = torch.real(self.model(torch.conj(self.encoded), action) / self.D)
        return output

    """
    Yang's version
    def feedback(self, indexes, r_pred, r_true, action):
        for i, r_p, r in zip(indexes, r_pred, r_true):
            encoded = self.encoded[i]
            temp = torch.outer(action, encoded)
            self.model += self.lr * (r - r_p) * temp
    """

    def feedback(self, loss, action: torch.Tensor):
        encoded = self.encoded
        temp = torch.outer(action, encoded)
        with torch.no_grad():
            self.model.l1.weight += self.lr * loss * temp

    def copy_param(self, net, initial=False):
        self.model = deepcopy(net.model)
        # TODO: seperate encoder and nn
        """
        if initial == True:
            self.s_hdvec = deepcopy(net.s_hdvec)
            self.bias = deepcopy(net.bias)
        """

