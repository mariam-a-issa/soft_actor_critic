import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd.functional import hessian
from copy import deepcopy
import os
import gym


GAMMA_ACTOR = .003
GAMMA_CRITIC = .009
TRACE_DECAY = .8
C = 1.2 #Some constant greater than 0

class Actor(nn.Module):

    def __init__(self, input_size : int, output_size : int, save = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._hidden_size = 24
        self.features = nn.Sequential(
                                    nn.Linear(input_size, self._hidden_size),
                                   # nn.ReLU(),
                                    #nn.Linear(self.hidden_size, self.parameter_size),
                                    nn.ReLU(),
                                    nn.Linear(self._hidden_size, output_size))
        
        self.parameter_size = len(self.score_function(torch.rand(self.input_size))[0])
        self._save = save
        
    def forward(self, x) -> torch.Tensor:
        probs =  nn.functional.softmax(self.features(x), dim=0)
        return probs + torch.full((self.output_size,), .001) #Add small value so never zero
    
    def score_function(self, state : torch.Tensor):
        """Calculate psi w.r.t the parameters of the nn.
        Each index of the returned tensor corresponds to the action."""
        grad_reset = deepcopy(optim.Adam(self.parameters())) #Will be used only to reset the gradients in the network
        
        output = torch.log(self(state))

        score_functions = torch.Tensor([])
        for i in range(self.output_size):
            grad_reset.zero_grad()
            output[i].backward(retain_graph=True)
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


class Critic:

    def __init__(self, actor : Actor) -> None:
        self.actor = actor
        self.output_size = actor.output_size
        self.feature_size = actor.parameter_size + actor.input_size ** 2 #Will be the length of the actors parameters plus the length of the flattened hessian
        self.parameters = torch.rand(self.feature_size)
        self.estimate_average_cost = 0
        self.trace = EligibilityTrace(self)

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

    def update_parameters(self, td : float) -> None:
        """Will update the paramters of the critic according to papers formula"""
        self.parameters = self.parameters + GAMMA_CRITIC * td * self.trace.get_trace()

    def update_average_cost(self, cost : float) -> None:
        """The given cost is given by the environment"""
        self.estimate_average_cost = self.estimate_average_cost + GAMMA_CRITIC * (cost - self.estimate_average_cost)

    def update_trace(self, state : torch.Tensor, action : int) -> None:
        "Should be done at the end of each episode"
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
        self._actor_optimizer = optim.Adam(self._actor.parameters(), GAMMA_ACTOR)
        self._gm = gm
        self._num_critic_updates = 0

    def get_state(self) -> torch.Tensor:
        """Get state from environment"""
        return torch.Tensor(self._gm.get_state())
    
    def get_action(self, state : torch.Tensor) -> int:
        """Returns an integer which will correspond to what action to take"""
        #nan is in network due to backprop?
        probs = self._actor(state)
        m = Categorical(probs)
        return int(m.sample())
    
    def update(self, state : torch.Tensor, action : int, next_state : torch.Tensor, next_action : int, reward : float) -> None:
        """Will update the actor and critic parameters"""
        q_next = self._critic.forward(next_state)
        q_current = self._critic.forward(state)
        cost =  -1 * reward

        #Critic update
        td = float(cost - self._critic.estimate_average_cost + q_next[next_action] - q_current[action])

        self._critic.update_parameters(td)
        self._num_critic_updates += 1
        
        self._critic.update_average_cost(cost)
        self._critic.update_trace(next_state, next_action)
        
        #Actor update
        t = C / (1 + self._critic.parameters.norm())            

        total_cost = t * float(q_next[next_action]) * torch.log(self._actor(next_state)[next_action]) #TODO When model gives prob as 0, it results in nan values to be propogated in network
        self._actor_optimizer.zero_grad()
        total_cost.backward()
        self._actor_optimizer.step()

        self._num_critic_updates = 0
    
    def reset(self) -> torch.Tensor:
        """Resets the environment and other parameters of agent and returns the new start state"""
        state =self._gm.reset()
        self._critic.reset()
        self._actor.save('train')
        return torch.Tensor(state[0])
    

def train(agent : Agent, *, epochs : int = None) -> Agent:
    state = agent.reset()
    action = agent.get_action(state)
    episodes = 0

    while True:
        next_state, reward, done, _, _ = agent._gm.step(action)
        next_state = torch.Tensor(next_state)
        new_action = agent.get_action(torch.Tensor(next_state))

        agent.update(state, action, next_state, new_action, reward)

        if done:
            state = agent.reset()
            agent.reset()
            print(episodes)
            episodes += 1
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