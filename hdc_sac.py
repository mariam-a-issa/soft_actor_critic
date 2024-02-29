import torch
from torch import Tensor
from collections import namedtuple
from math import pi

HYPER_DIM = 8192 #The dimensionality of the hypervectors
V_LR = .001
Q_LR = .001
TAU = .005
DISCOUNT = .99


Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'next_action', 'reward'])

class EXPEncoder:
    """Represents the exponential encoder from the hdpg_actor_critic code but changed to only use a tensors from pytorch"""
    def __init__(self, d : int):
        """Will create an encoder that will work for vectors that have d dimensionality"""
        super(EXPEncoder, self).__init__()
        
        self._s_hdvec = Tensor(HYPER_DIM).uniform_(0, 1).unsqueeze(0)
        for _ in range(d - 1):
            self._s_hdvec = torch.cat((self._s_hdvec, Tensor(HYPER_DIM).uniform_(0, 1).unsqueeze(0)), dim = 0)
        
        self._bias = Tensor(HYPER_DIM).uniform_(0, 2 * pi)
        self.d = d #Will be used by functions to know how to create their models

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the encoder hypervector. State needs the same dimensionality that was used to create the encoder"""
        return torch.exp(1j * (torch.matmul(state, self._s_hdvec) + self._bias))


class HDModel:
    """Represents a base class for a a hyperdimensional model that using the equation 5 in the hdpg paper"""
    def __init__(self) -> None:
        """Will create a model which"""
        temp_d = 32 #Could be a potential hyper parameter?
        self._m_hdvec = EXPEncoder(temp_d)(Tensor(temp_d).uniform_(0, 1)) #Will need to find a way to initilze these parameters. Could use nn.Linear but that does not complex numbers which I think it needs?

    def __call__(self, x : Tensor) -> Tensor:
        """Will return a single value off of the vector x which should have a dimensioanilty of HYPER_DIM"""
        return torch.real((torch.conj(x) * self._m_hdvec) / HYPER_DIM)
    
    def params(self) -> Tensor:
        """Will return the parameters of the network which is just the hypervector"""
        return self._m_hdvec
    
    def update_params(self, u : Tensor) -> None:
        """Will set the vector u as the new params of the network"""
        self._m_hdvec = u
        


class QModel(HDModel):
    def __init__(self, state_en : EXPEncoder, action_en : EXPEncoder, target_v : 'TargetValueFunction') -> None:
        super().__init__()
        self._t_v = target_v
        self._state_en = state_en
        self._action_en = action_en

    def __call__(self, state : Tensor, action : Tensor) -> Tensor:
        """Will create a hypervector that represents both the state and the action and then call the model bassed off of this.
        The state and action should not be encoded and have their respective dimensions"""
        state_action_hdvec = torch.add(self._state_en(state), self._action_en(action))
        return super().__call__(state_action_hdvec)
    
    def update(self, trans : Transition) -> Tensor:
        """Will update the q functions using equation 9 and algorithim 1 of the SAC paper"""
        new_params = self.params() - Q_LR * (self(trans.state, trans.action) - trans.reward + self._t_v(trans.next_state)) * self.params()
        self.update_params(new_params)


class QFunction:
    def __init__(self, state_en : EXPEncoder, action_en : EXPEncoder, target_v : 'TargetValueFunction') -> None:
        self._q1 = QModel(state_en, action_en, target_v)
        self._q2 = QModel(state_en, action_en, target_v)

    def __call__(self, state : Tensor, action : Tensor) -> Tensor:
        """Will return the q function that has the minimum value according to the SAC paper"""
        return torch.min(self._q1(state, action), self._q2(state, action))
    
    def update(self, trans : Transition) -> None:
        """Will update the both of the q models in the q function"""
        self._q1.update(trans)
        self._q2.update(trans)

class ValueFunction(HDModel):
    def __init__(self, encoder : EXPEncoder, q_func : QFunction) ->None:
        super().__init__()
        self.encoder = encoder
        self._q_func = q_func

    def __call__(self, state : Tensor) -> Tensor:
        """Will return the value of the state which is a vector of state dimensioanlity"""
        return super().__call__(self.encoder(state))

    def update(self, trans : Transition) -> None:
        """Updates the value function using equation 6 and algorithim 1 on the SAC paper"""
        new_params = self.params() - V_LR * (self(trans.state) - self._q_func(trans.state, trans.action)) * self.params()
        self.update_params(new_params)

class TargetValueFunction(ValueFunction):
    def __init__(self, encoder : EXPEncoder, v_func : ValueFunction):
        super().__init__(encoder, None)
        self._v_func = v_func

    def update(self, trans: Transition) -> None:
        """Will update the function according to equation in algorithm 1 of the SAC paper"""
        self.update_params(TAU * self._v_func.params() + (1 - TAU) * self.params())




