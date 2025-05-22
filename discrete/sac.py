import math

import torch
from torch import Tensor

class Alpha:
    """
    Implements an alpha parameter for entropy regularization in reinforcement learning, with optional auto-tuning.
    """
    
    def __init__(self,
                 start: float,
                 end: float,
                 midpoint: float,
                 slope: float,
                 max_steps: int,
                 autotune: bool = False,
                 alpha_value: float = None):
        """
        Initializes the Alpha class.
        
        Args:
            start (float): Starting target normalized entropy.
            end (float): Ending target normalized entropy.
            midpoint (float): Midpoint of the sigmoid decay of the target entropy.
            slope (float): Slope of the sigmoid decay.
            max_steps (int): Total number of training steps.
            autotune (bool, optional): If True, start is used as the initial alpha value. Defaults to False.
            alpha_value (float, optional): Fixed alpha value if autotune is False. Defaults to None.
        """
        self._start = start
        self._end = end
        self._midpoint = midpoint
        self._slope = slope
        self._log_alpha = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        self._max_steps = max_steps
        self._autotune = autotune
        self._alpha_value = alpha_value
        self._current_step = 0
        
    def __call__(self) -> Tensor:
        """
        Returns the current alpha value.
        
        Returns:
            Tensor: The current alpha value.
        """
        if not self._autotune:
            return torch.tensor(self._alpha_value, device='cpu')
        return self._log_alpha.exp()
    
    def sigmoid_target_entropy(self) -> float:
        """
        Computes the target entropy using a sigmoid decay function.
        
        Returns:
            float: The current target entropy value.
        """
        self._current_step += 1
        x = self._current_step / self._max_steps  # Normalize step to [0,1]
        return self._start + (self._end - self._start) / (1 + math.exp(-self._slope * (x - self._midpoint)))
    
    def parameters(self) -> list[Tensor]:
        """
        Returns the trainable alpha parameter.
        
        Returns:
            list[Tensor]: A list containing the log alpha parameter.
        """
        return [self._log_alpha]
    
    def to(self, device: torch.device) -> None:
        """
        Moves the alpha parameter to the specified device.
        
        Args:
            device (torch.device): The target device.
        """
        self._log_alpha.to(device)

def policy_loss(q_target: Tensor,
                action_probs: Tensor,
                action_log_probs: Tensor,
                alpha: float) -> Tensor:
    """
    Computes the policy loss.
    
    Args:
        q_target (Tensor): Q-values for each action (batch_size x action_size).
        action_probs (Tensor): Action probabilities (batch_size x action_size).
        action_log_probs (Tensor): Log probabilities of actions (batch_size x action_size).
        alpha (float): Entropy regularization coefficient.
    
    Returns:
        Tensor: Policy loss (batch_size x 1).
    """
    batch_size, action_size = q_target.shape
    difference = alpha * action_log_probs - q_target
    loss = torch.bmm(action_probs.view(batch_size, 1, action_size), difference.view(batch_size, action_size, 1)).view(batch_size, 1)
    return loss

def q_func_loss(cur_q1: Tensor,
                cur_q2: Tensor,
                next_q_target: Tensor,
                cur_action: Tensor,
                next_action_probs: Tensor,
                next_action_log_probs: Tensor,
                reward: Tensor,
                alpha: float,
                discount: float,
                done: Tensor) -> tuple[Tensor, Tensor]:
    """
    Computes the loss for two Q-functions using temporal difference learning.
    
    Args:
        cur_q1 (Tensor): Current Q-values from the first Q-network (batch_size x action_size).
        cur_q2 (Tensor): Current Q-values from the second Q-network (batch_size x action_size).
        next_q_target (Tensor): Target Q-values for the next state (batch_size x action_size).
        cur_action (Tensor): Selected actions (batch_size x 1).
        next_action_probs (Tensor): Action probabilities for the next state (batch_size x action_size).
        next_action_log_probs (Tensor): Log action probabilities for the next state (batch_size x action_size).
        reward (Tensor): Reward signal (batch_size x 1).
        alpha (float): Entropy regularization coefficient.
        discount (float): Discount factor.
        done (Tensor): Boolean tensor indicating terminal states (batch_size x 1).
    
    Returns:
        tuple[Tensor, Tensor]: Difference between actual and predicted Q-values for both Q-networks (batch_size x 1 each).
    """
    q_log_dif = next_q_target - alpha * next_action_log_probs
    batch_size, action_size = next_action_probs.shape
    next_v = torch.bmm(next_action_probs.view(batch_size, 1, action_size), q_log_dif.view(batch_size, action_size, 1)).view(batch_size, 1)
    actual_q = reward + (1 - done) * discount * next_v
    q1_a = cur_q1.gather(1, cur_action)
    q2_a = cur_q2.gather(1, cur_action)
    return actual_q - q1_a, actual_q - q2_a

def alpha_loss(probs: Tensor,
               log_probs: Tensor,
               alpha: Tensor,
               target_entropy: Tensor) -> Tensor:
    """
    Computes the loss for updating the alpha parameter.
    
    Args:
        probs (Tensor): Action probabilities (batch_size x action_size).
        log_probs (Tensor): Log action probabilities (batch_size x action_size).
        alpha (Tensor): Entropy regularization coefficient.
        target_entropy (Tensor): Target entropy value.
    
    Returns:
        Tensor: Alpha loss.
    """
    batch_size, action_size = probs.shape
    return torch.bmm(probs.detach().view(batch_size, 1, action_size), (-alpha * (log_probs + target_entropy).detach()).view(batch_size, action_size, 1)).mean()

def mse(input: Tensor) -> Tensor:
    """
    Computes the mean squared error (MSE) loss.
    
    Args:
        input (Tensor): Differences between actual and predicted values (batch_size x 1).
    
    Returns:
        Tensor: MSE loss.
    """
    return (1/2 * (input ** 2)).mean().squeeze()