import torch
from torch import Tensor

def policy_loss(q_target : Tensor,
                action_probs : Tensor,
                action_log_probs : Tensor,
                alpha : float):
    """Will calcuate the loss for the policy
        q_target : batch_size x action 
        action_probs : batch_size x action 
        action_log_probs : batch_size x action
        
        return
        loss as a batch_size x 1 tensor
    """
    
    batch_size, action_size = q_target.shape
    difference = alpha * action_log_probs - q_target
    loss = torch.bmm(action_probs.view(batch_size, 1, action_size), difference.view(batch_size, action_size, 1)).view(batch_size, 1)
    
    return loss


def q_func_loss(cur_q1 : Tensor,
                cur_q2 : Tensor,
                next_q_target : Tensor,
                cur_action : Tensor,
                next_action_probs : Tensor,
                next_action_log_probs : Tensor,
                reward : Tensor,
                alpha : float,
                discount : float,
                done : Tensor) -> tuple[Tensor, Tensor]:
    """Will calculate the loss(simple difference) for the two q functions
    
        q1_cur, q2_cur: batch_size x action tensor for current q values
        q_target_next : batch_size x action tensor for next state target q values
        action : batch_size x 1 tensor for choosen action
        next_action_probs : batch_size x action tensor for probability of choosing action given next state
        next_action_log_probs : batch_size x action tensor same as next_action_probs but log
        reward : batch_size x 1 tensor 
        alpha : The alpha term controlling entropy
        discount : Discount term 
        done : batch_size x 1 tensor containing boolean values on if the current state is the final state
        
        returns
        Q1(actual) - Q1(current), Q2(actual) - Q2(current) each a bx1 tensor
        Done this way due to how NN may do MSE wheras HDC will directly use this value
    """
    
    q_log_dif = next_q_target - alpha * next_action_log_probs
    
    batch_size, action_size = next_action_probs.shape
    
    #Essentially batch dot product
    next_v = torch.bmm(next_action_probs.view(batch_size, 1, action_size),
                       q_log_dif.view(batch_size, action_size, 1)).view(batch_size, 1)
    
    actual_q = reward + (1 - done) * discount * next_v
    
    q1_a = cur_q1.gather(1, cur_action)
    q2_a = cur_q2.gather(1, cur_action)
    
    return actual_q - q1_a, actual_q - q2_a
    
    
    
def mse(input : Tensor) -> Tensor:
    """Reduces a batched tensor to a single loss
        input: bx1 tensor each element is a difference
        
        returns MSE
    """
    
    return  (1/2*(input ** 2)).mean().squeeze()