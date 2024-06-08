import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Categorical
from torch.autograd.functional import hessian
from copy import deepcopy
import os
import gym
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


ACTOR_START_GAMMA = .8
CRITIC_START_GAMMA = .8

#For both decays .5 < decay < 1
#And critic decay < actor decay
ACTOR_DECAY = .9
CRITIC_DECAY = .8

ACTOR_GAMMA = lambda epoch: 1 / ((epoch + 1) ** ACTOR_DECAY)
CRITIC_GAMMA = lambda epoch: 1/ ((epoch + 1) ** CRITIC_DECAY)

TRACE_DECAY = .9
C = 1.2 #Some constant greater than 0

class Gamma:

    def __init__(self, start : float, update : 'update_func') -> None:
        self._iterations = 0
        self._start = start
        self._update = update

    def __call__(self) -> float:
        gamma = self._start * self._update(self._iterations)
        self._iterations += 1
        return gamma


class Actor(nn.Module):

    def __init__(self, input_size : int, output_size : int, save = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._hidden_size = 24
        self.features = nn.Sequential(
                                    nn.Linear(input_size, self._hidden_size),
                                    #nn.ReLU(),
                                    #nn.Linear(self.hidden_size, self.parameter_size),
                                    nn.ReLU(),
                                    nn.Linear(self._hidden_size, output_size))
        
        self.parameter_size = len(self.score_function(torch.rand(self.input_size))[0])
        self._save = save
        
    def forward(self, state : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        _, dist = self._prob_distribution(state)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
        
        
    
    def score_function(self, state : torch.Tensor):
        """Calculate psi w.r.t the parameters of the nn.
        Each index of the returned tensor corresponds to the action."""
        grad_reset = deepcopy(optim.SGD(self.parameters())) #Will be used only to reset the gradients in the network
        
        _, dist = self._prob_distribution(state)

        log_probs = torch.Tensor([])
        for i in range(self.output_size):
            log_probs = torch.cat((log_probs, dist.log_prob(torch.Tensor([i]))), dim = 0)

        score_functions = torch.Tensor([])
        for i in range(self.output_size):
            grad_reset.zero_grad()
            log_probs[i].backward(retain_graph=True)
            gradients = torch.Tensor([])

            for param in self.parameters():
                gradients = torch.cat((gradients, param.grad.flatten(start_dim=0)), dim = 0)

            score_functions = torch.cat((score_functions, deepcopy(gradients.unsqueeze(0))), dim = 0)

        return score_functions
    
    def calculate_hessian(self, state : torch.Tensor) -> torch.Tensor:
        """Returns a flattened hessian w.r.t to the input.
        Each index corresponds to an action."""
        hessians = torch.Tensor([])

        for i in range(self.output_size):

            def hessian_for_output(x):
                output = self(x)
                return output[i]
            
            h = torch.flatten(hessian(hessian_for_output, state), start_dim=0).unsqueeze(0)
            hessians = torch.cat((hessians, h), dim = 0)
        
        return hessians
    
    def save(self, type, file_name='best_weights.pt') -> None:
        """Will save the model in the folder 'model' in the dir that the script was run in."""
        if not self._save:
            return

        model_folder_path = './model/' + str(type)

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_name)

    def _prob_distribution(self, state : torch.Tensor) -> tuple[torch.Tensor, Categorical]:
        """Returns a a tensor of the probabilites and its distribution"""

        probs =  nn.functional.softmax(self.features(state), dim=0)
        dist = Categorical(probs)

        return probs, dist

        


class Critic:

    def __init__(self, actor : Actor) -> None:
        self.actor = actor
        self.output_size = actor.output_size
        self.feature_size = actor.parameter_size + actor.input_size ** 2 #Will be the length of the actors parameters plus the length of the flattened hessian
        self.parameters = torch.rand(self.feature_size)
        self.estimate_average_cost = 0
        self.trace = EligibilityTrace(self)
        self._gamma = Gamma(CRITIC_START_GAMMA, CRITIC_GAMMA)

    def features(self, state : torch.Tensor) -> torch.Tensor:
        """Returns the features of the critic which include psi (actors score function).
        Additional features about the actors hessian w.r.t to the input are added in order to help the critic better approximate the q funciton.
        Each index of the returned tensor corresponds to an action."""
        return torch.cat((self.actor.calculate_hessian(state), self.actor.score_function(state)), dim=1)
    
    #TODO could implement as a nn.Parameter instead of a serperate Tensor
    def forward(self, state : torch.Tensor) -> torch.Tensor:
        """Returns tensor where each index corresponds to the q value for that action"""
        q_tensor = torch.Tensor([]) #Will contain a tensor of all of the q value for each action given a state
        
        for i in range(self.output_size):
            q = torch.dot(self.parameters, self.features(state)[i]).unsqueeze(0)
            q_tensor = torch.cat((q_tensor, q))

        return q_tensor

    def update_parameters(self, td : float, state : torch.Tensor, action : int, cost : float) -> None:
        """Will update the paramters of the critic according to papers formulas"""
        cur_gamma = self._gamma()
        self.parameters = self.parameters + cur_gamma * td * self.trace.get_trace()
        self.estimate_average_cost = self.estimate_average_cost + cur_gamma * (cost - self.estimate_average_cost)
        self.trace.update_trace(state, action)

    def reset(self) -> None:
        """Reset the rewards and the trace at the end of each episode"""
        self.estimate_average_cost = 0
        self.trace.reset_trace()
    

class EligibilityTrace:

    def __init__(self, critic : Critic) -> None:
        self._critic = critic
        self._trace = torch.zeros(critic.feature_size)

    def update_trace(self, state : torch.Tensor, action : int) -> None:
        """Will update the trace according to trace decay equation in paper"""
        self._trace = TRACE_DECAY * self._trace + self._critic.features(state)[action]

    def get_trace(self) -> torch.Tensor:
        return self._trace

    def reset_trace(self) -> None:
        """Resets the trace which should be done after each episode"""
        self._trace.zero_()


class Agent:

    def __init__(self, gm : gym.Env, input_size : int, output_size : int) -> None:
        self._actor = Actor(input_size, output_size)
        self._critic = Critic(self._actor)
        self._actor_optimizer = optim.SGD(self._actor.parameters(), ACTOR_START_GAMMA)
        self._lr_scheduluar = LambdaLR(self._actor_optimizer, ACTOR_GAMMA)
        self._gm = gm
        self._updates = 0

    def get_state(self) -> torch.Tensor:
        """Get state from environment"""
        return torch.Tensor(self._gm.get_state())
    
    def get_action(self, state : torch.Tensor) -> tuple[int, torch.Tensor]:
        """Returns an integer which will correspond to what action to take and the log probability of the action"""
        #nan is in network due to backprop?

        action, log_prob = self._actor(state)
        
        return int(action), log_prob
    
    def update(self, next_log_prob : torch.Tensor, state : torch.Tensor, action : int, next_state : torch.Tensor, next_action : int, reward : float) -> None:
        """Will update the actor and critic parameters"""
        q_next = self._critic.forward(next_state)
        q_current = self._critic.forward(state)
        cost =  -1 * reward

        #Critic update
        td = float(cost - self._critic.estimate_average_cost + q_next[next_action] - q_current[action])
        self._critic.update_parameters(td, next_state, next_action, cost)
        writer.add_scalar('Critic td', td, self._updates)

        #Actor update
        t = C / (1 + self._critic.parameters.norm())            

        total_cost = t * float(q_next[next_action]) * next_log_prob #TODO When model gives prob as 0, it results in nan values to be propogated in network
        
        writer.add_scalar('Actor total cost', total_cost, self._updates)
        
        self._actor_optimizer.zero_grad()
        total_cost.backward()
        self._actor_optimizer.step()

        self._lr_scheduluar.step()
        writer.flush()
        self._updates += 1

    def reset(self) -> torch.Tensor:
        """Resets the environment and other parameters of agent and returns the new start state"""
        state =self._gm.reset()
        self._critic.reset()
        self._actor.save('train')
        return torch.Tensor(state[0])
    

def train(agent : Agent, *, epochs : int = None) -> Agent:
    state = agent.reset()
    action, _ = agent.get_action(state)
    episodes = 0

    total_reward = 0
    while True:
        next_state, reward, done, _, _ = agent._gm.step(action)
        next_state = torch.Tensor(next_state)
        new_action, next_log_prob = agent.get_action(torch.Tensor(next_state))

        total_reward += reward

        agent.update(next_log_prob, state, action, next_state, new_action, reward)

        if done:
            writer.add_scalar('Total Reward', total_reward, episodes)
            writer.flush()
            print(episodes)

            state = agent.reset()
            episodes += 1
            total_reward = 0
            action, _ = agent.get_action(state)
            if episodes == epochs:
                return agent

        state = next_state
        action = new_action 

def evaluate(agent : Agent) -> None:
    state = agent.reset()
    action = agent.get_action(state)

    while True:
        next_state, _, done, _, _ = agent._gm.step(action)
        next_state = torch.Tensor(next_state)
        new_action = agent.get_action(torch.Tensor(next_state))

        if done:
            state = agent.reset()

        state = next_state
        action = new_action 