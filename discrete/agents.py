from abc import ABC, abstractmethod
from typing import Iterable

from torch import Tensor
from torch_geometric.data import Data

from utils import Transition, LearningLogger

# def create_mil_hdc_agent(device_size : int,
#                  action_size : int,
#                  hyper_dim : int,
#                  policy_lr : float,
#                  critic_lr : float,
#                  alpha_lr : float,
#                  discount : float,
#                  tau : float,
#                  alpha_value : float,
#                  target_update : int, #When the target should update
#                  update_frequency : int, #When the models should update,
#                  learning_steps : int, #Amount of gradient steps
#                  device : torch.device,
#                  buffer_length : int,
#                  sample_size : int,
#                  clip_norm_value : float,
#                  target_ent_start : float,
#                  target_ent_end : float,
#                  alpha_slope : float,
#                  midpoint : float,
#                  max_steps : float,
#                  autotune : bool,
#                  random : random):
    
#     embed = mil_hdc.Encoder(hyper_dim, device_size)
#     policy = mil_hdc.Actor(hyper_dim, action_size)
#     q_func = mil_hdc.QFunction(hyper_dim, action_size)
#     q_target = mil_hdc.QFunctionTarget(q_func, tau)
#     alpha = mil_nn.Alpha(target_ent_start, target_ent_end, midpoint, alpha_slope, max_steps, autotune, alpha_value)
    
#     optim_policy = torch.optim.Adam(policy.parameters(), policy_lr)
#     optim_alpha = torch.optim.Adam([alpha._log_alpha], alpha_lr)
    
#     memory = DynamicMemoryBuffer(buffer_length, sample_size, random)
    
#     for obj in [embed, policy, q_func, q_target, alpha]:
#         obj.to(device)
        
#     TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
#     def trace_handler(prof: torch.profiler.profile):
#         # Prefix for file names.
#         host_name = socket.gethostname()
#         timestamp = datetime.now().strftime(TIME_FORMAT_STR)
#         file_prefix = f"{host_name}_{timestamp}"

#         # Construct the trace file.
#         prof.export_chrome_trace(f"{file_prefix}.json.gz")

#         # Construct the memory timeline file.
#         prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    
#     with torch.profiler.profile(
#        activities=[
#            torch.profiler.ProfilerActivity.CPU,
#            torch.profiler.ProfilerActivity.CUDA,
#        ],
#        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
#        record_shapes=True,
#        profile_memory=True,
#        with_stack=True,
#        on_trace_ready=trace_handler
#    ) as prof:
        
#         def update(steps) -> Tensor:
#             """Will return tensor of the losses in a tensor of dim 6 in order of Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
#             prof.step()
            
#             trans = memory.sample()
            
#             with record_function('## forward ##'):
#                 cur_state, cur_batch_index = embed(trans.state, trans.state_index)
                
#                 cur_q1, cur_q2 = q_func(cur_state, cur_batch_index, trans.state_index)
#                 _, cur_prob, cur_log_prob = policy.sample_action(cur_state, cur_batch_index, trans.state_index)
                
#                 number_devices = torch.diff(trans.state_index)
#                 cur_log_prob = cur_log_prob / torch.log(number_devices * action_size).view(-1, 1)
                
#                 with torch.no_grad():
#                     cur_q_target = q_target(cur_state, cur_batch_index, trans.state_index)
                    
#                     next_state_embed, next_batch_index = embed(trans.next_state, trans.next_state_index)
                    
#                     next_q_target = q_target(next_state_embed, next_batch_index, trans.next_state_index)
#                     _, next_prob, next_log_prob = policy.sample_action(next_state_embed, next_batch_index, trans.next_state_index)
                    
#                     next_log_prob = next_log_prob / torch.log(number_devices * action_size).view(-1, 1) #Normilize by the maximum possible entropy
#                     batch_size, cur_action_size = cur_prob.shape
                    
#                     ent = -torch.bmm(cur_prob.view(batch_size, 1, cur_action_size),
#                                     cur_log_prob.view(batch_size, cur_action_size, 1)).mean()
            
#             with record_function('### Loss Calc ###'):  
#                 policy_loss = sac.policy_loss(cur_q_target, cur_prob, cur_log_prob, alpha()).mean().squeeze()
#                 q1_dif, q2_dif = sac.q_func_loss(cur_q1, 
#                                                 cur_q2,
#                                                 next_q_target,
#                                                 trans.action,
#                                                 next_prob,
#                                                 next_log_prob,
#                                                 trans.reward,
#                                                 alpha(),
#                                                 discount,
#                                                 trans.done)
#                 alpha_loss = sac.alpha_loss(cur_prob,
#                                             cur_log_prob,
#                                             alpha(),
#                                             alpha.sigmoid_target_entropy(steps))
            
#             #Need to get the vector corresponding to the chosen action which would involve doing integer divison between the action index and the total number of devices in a state    
#             #Find what devices correspond to the choosen action and then select only the device embedding for that action
            
#             with record_function('##Backward Pass##'):
#                 device_choosen = trans.action.squeeze() // action_size
#                 indexes = trans.state_index[:-1] + device_choosen
                
#                 matrix_l1 = q1_dif * cur_state[indexes] * critic_lr
#                 matrix_l2 = q2_dif * cur_state[indexes] * critic_lr
                
#                 q1_params : Tensor
#                 q2_params : Tensor
#                 q1_params, q2_params = q_func.parameters()
                
#                 optim_policy.zero_grad()
#                 policy_loss.backward()
#                 optim_policy.step()
                
#                 #Need to mod due to only a single model but multiple devices
#                 q1_params.index_add_(0, trans.action.squeeze() % action_size, matrix_l1)
#                 q2_params.index_add_(0, trans.action.squeeze() % action_size, matrix_l2)

#                 optim_alpha.zero_grad()
#                 alpha_loss.backward()
                
#                 optim_policy.step()
#                 optim_alpha.step()
                
#                 q1_loss = sac.mse(q1_dif)
#                 q2_loss = sac.mse(q2_dif )
            
#             return torch.tensor([
#                 q1_loss,
#                 q2_loss,
#                 policy_loss,
#                 ent,
#                 alpha_loss,
#                 alpha()
#             ]).to(device)

#     def call(state : Tensor | Data) -> Tensor:
#         with torch.no_grad():
#             state_index = torch.tensor([0,state.shape[0]])
#             embed_state, batch_index = embed(state, state_index)
#             action, _, _ = policy.sample_action(embed_state, batch_index, state_index)
#             return action
        
#     def evaluate(state : Tensor | Data) -> Tensor:
#         with torch.no_grad():
#             state_index = torch.tensor([0,state.shape[0]])
#             embed_state, batch_index = embed(state, state_index)
#             action = policy.evaluate_action(embed_state, batch_index, state_index)
#             return action
        
#     def add_data(trans : Transition) -> None:
#         memory.add_data(trans)
        
#     return Agent(
#         target_update,
#         update_frequency,
#         learning_steps,
#         update,
#         call,
#         q_target.update,
#         lambda x : None,
#         evaluate,
#         add_data
#     )
        

# def create_mil_nn_agent(device_size : int,
#                  action_size : int,
#                  embed_size : int,
#                  pos_encode_size : int,
#                  policy_lr : float,
#                  critic_lr : float,
#                  alpha_lr : float,
#                  discount : float,
#                  tau : float,
#                  alpha_value : float,
#                  target_update : int, #When the target should update
#                  update_frequency : int, #When the models should update,
#                  learning_steps : int, #Amount of gradient steps
#                  device : torch.device,
#                  buffer_length : int,
#                  sample_size : int,
#                  clip_norm_value : float,
#                  target_ent_start : float,
#                  target_ent_end : float,
#                  alpha_slope : float,
#                  midpoint : float,
#                  max_steps : float,
#                  autotune : bool,
#                  random : random,
#                  attention : bool,
#                  num_heads : int,
#                  graph : bool,
#                  message_passed : int):
    
#     #When creating the embedding functions understand the output sizes of the embeddings.
#     if attention:
#         q_embedding = mil_nn.AttentionEmbedding(embed_size, pos_encode_size, device_size, num_heads)
#         target_q_embedding = deepcopy(q_embedding)
#         policy_embedding = mil_nn.AttentionEmbedding(embed_size, pos_encode_size, device_size, num_heads)
#         new_embed_size = embed_size
        
#     elif graph:
#         q_embedding = mil_nn.GraphEmbedding(embed_size, pos_encode_size, device_size, message_passed)
#         target_q_embedding = deepcopy(q_embedding)
#         policy_embedding = mil_nn.GraphEmbedding(embed_size, pos_encode_size, device_size, message_passed)
#         new_embed_size = embed_size
#     else:
#         q_embedding = mil_nn.Embedding(embed_size, pos_encode_size, device_size)
#         target_q_embedding = deepcopy(q_embedding)
#         policy_embedding = mil_nn.Embedding(embed_size, pos_encode_size, device_size)
        
#     q_func = mil_nn.QFunction(new_embed_size, action_size)
#     q_func_target = mil_nn.QFunctionTarget(q_func, tau)
#     policy = mil_nn.Actor(new_embed_size, action_size)
#     alpha = mil_nn.Alpha(target_ent_start, target_ent_end, midpoint, alpha_slope, max_steps, autotune, alpha_value)
    
#     optim_critic = torch.optim.Adam([*q_embedding.parameters(), *q_func.parameters()], lr=critic_lr)
#     optim_policy = torch.optim.Adam([*policy_embedding.parameters(), *policy.parameters()], lr=policy_lr)
#     optim_alpha = torch.optim.Adam([alpha._log_alpha], lr = alpha_lr)
    
#     if graph:
#         memory = GraphMemoryBuffer(buffer_length, sample_size, random)
#     else:
#         memory = DynamicMemoryBuffer(buffer_length, sample_size, random)
    
#     for obj in [q_embedding, policy_embedding, q_func, q_func_target, policy, alpha]:
#         obj.to(device)
    
#     def update(steps) -> Tensor:
#         """Will return tensor of the losses in a tensor of dim 6 in order of Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
        
#         trans = memory.sample()
        
#         q_cur_state_embed, q_cur_batch_index = q_embedding(trans.state, trans.state_index)
#         policy_cur_state_embed, policy_cur_batch_index = policy_embedding(trans.state, trans.state_index)
        
#         cur_q1, cur_q2 = q_func(q_cur_state_embed, q_cur_batch_index, trans.state_index)
#         _, cur_prob, cur_log_prob = policy.sample_action(policy_cur_state_embed, policy_cur_batch_index)
        
#         number_devices = torch.diff(trans.state_index)
#         cur_log_prob = cur_log_prob / torch.log(number_devices * action_size).view(-1, 1) #Normilize by the maximum possible entropy
        
#         with torch.no_grad():
#             tar_q_cur_state_embed, tar_q_cur_batch_index = target_q_embedding(trans.state, trans.state_index)
#             cur_q_target = q_func_target(tar_q_cur_state_embed, tar_q_cur_batch_index, trans.state_index)
            
#             q_next_state_embed, q_next_batch_index = target_q_embedding(trans.next_state, trans.next_state_index)
#             policy_next_state_embed, policy_next_batch_index = policy_embedding(trans.next_state, trans.next_state_index)
            
#             next_q_target = q_func_target(q_next_state_embed, q_next_batch_index, trans.next_state_index)
#             _, next_prob, next_log_prob = policy.sample_action(policy_next_state_embed, policy_next_batch_index)
            
#             next_log_prob = next_log_prob / torch.log(number_devices * action_size).view(-1, 1) #Normilize by the maximum possible entropy
#             batch_size, cur_action_size = cur_prob.shape
            
#             ent = -torch.bmm(cur_prob.view(batch_size, 1, cur_action_size),
#                             cur_log_prob.view(batch_size, cur_action_size, 1)).mean()
        
        
#         policy_loss = sac.policy_loss(cur_q_target, cur_prob, cur_log_prob, alpha()).mean().squeeze()
#         q1_dif, q2_dif = sac.q_func_loss(cur_q1, 
#                                          cur_q2,
#                                          next_q_target,
#                                          trans.action,
#                                          next_prob,
#                                          next_log_prob,
#                                          trans.reward,
#                                          alpha(),
#                                          discount,
#                                          trans.done)
#         alpha_loss = sac.alpha_loss(cur_prob,
#                                     cur_log_prob,
#                                     alpha(),
#                                     alpha.sigmoid_target_entropy(steps))
        
#         q1_loss = sac.mse(q1_dif)
#         q2_loss = sac.mse(q2_dif)
        
#         optim_policy.zero_grad()
#         policy_loss.backward()
#         optim_policy.step()
        
#         critic_loss = q1_loss + q2_loss
        
#         optim_critic.zero_grad()
#         critic_loss.backward()
        
#         LearningLogger().log_scalars({'Grad of Policy' : _calc_grad_norm([*policy_embedding.parameters(), *policy.parameters()]),
#                                       'Unclipped Grad of Q Function' : _calc_grad_norm([*q_embedding.parameters(), *q_func.parameters()])},
#                                       steps=steps)
        
#         #utils.clip_grad_norm_([*q_embedding.parameters(), *q_func.parameters()], clip_norm_value)
        
#         optim_alpha.zero_grad()
#         alpha_loss.backward()
        
#         optim_policy.step()
#         optim_critic.step()
#         optim_alpha.step()
        
#         return torch.tensor([
#             q1_loss,
#             q2_loss,
#             policy_loss,
#             ent,
#             alpha_loss,
#             alpha()
#         ]).to(device)
        
#     def target_update_func():
#         q_func_target.update()
#         _polyak_average(q_embedding.parameters(), target_q_embedding.parameters(), tau)
        
#     def call(state : Tensor | Data) -> Tensor:
#         with torch.no_grad():
            
#             if graph:
#                 state = Batch.from_data_list([state])
#                 state_index = group_to_boundaries_torch(state.batch)
#             else:
#                 state_index = torch.tensor([0,state.shape[0]])
            
#             embed_state, batch_index = policy_embedding(state, state_index)
#             action, _, _ = policy.sample_action(embed_state, batch_index)
#             return action
        
#     def evaluate(state : Tensor | Data) -> Tensor:
#         with torch.no_grad():
            
#             if graph:
#                 state = Batch.from_data_list([state])
#                 state_index = group_to_boundaries_torch(state.batch)
#             else:
#                 state_index = torch.tensor([0,state.shape[0]])
            
#             embed_state, batch_index = policy_embedding(state, state_index)
#             action = policy.evaluate_action(embed_state, batch_index)
#             return action
        
#     def add_data(trans : Transition) -> None:
#         memory.add_data(trans)
        
#     return Agent(
#         target_update,
#         update_frequency,
#         learning_steps,
#         update,
#         call,
#         target_update_func,
#         lambda x : None,
#         evaluate,
#         add_data
#     )
            

# def create_nn_agent(input_size : int,
#                  output_size : int,
#                  hidden_size : int,
#                  policy_lr : float,
#                  critic_lr : float,
#                  alpha_lr : float,
#                  discount : float,
#                  tau : float,
#                  alpha_value : float,
#                  autotune : bool,
#                  target_update : int, #When the target should update
#                  update_frequency : int, #When the models should update,
#                  learning_steps : int, #Amount of gradient steps
#                  device : torch.device,
#                  dynamic : bool,
#                  grad_clip : float):
#     """Will create SAC agent based on NNs"""
    
#     if dynamic:
#         actual_input_size = input_size * MAX_ROWS
    
#     target_q = nn.QFunctionTarget(None, tau)
#     alpha = nn.Alpha(output_size, alpha_value, alpha_lr, autotune=autotune)

#     actor = nn.Actor(input_size, 
#                             output_size, 
#                             hidden_size,
#                             target_q,
#                             alpha, 
#                             policy_lr,
#                             dynamic,
#                             grad_clip)
        
#     q_function = nn.QFunction(input_size,
#                                      output_size,
#                                      hidden_size,
#                                      actor,
#                                      target_q,
#                                      alpha,
#                                      critic_lr,
#                                      discount,
#                                      dynamic,
#                                      grad_clip)
        
#     target_q.set_actual(q_function)
    
#     for obj in [target_q, alpha, actor, q_function]:
#         obj.to(device)
    
#     def update(trans : Transition) -> Tensor:
#         """Will return tensor of the losses in a tensor of dim 6 in order of Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
#         if dynamic:
#             trans = Transition(state = nn.pad(trans.state, actual_input_size),
#                        next_state = nn.pad(trans.next_state, actual_input_size),
#                        action=trans.action,
#                        reward=trans.reward,
#                        done=trans.done,
#                        num_devices=trans.num_devices,
#                        num_devices_n=trans.num_devices_n)
#         q_info = q_function.update(trans)
#         actor_info = actor.update(trans)
#         return torch.cat((q_info, actor_info))
    
#     def call(state : Tensor) -> Tensor:
#         """Will return the action that should be executed at the given state"""
#         with torch.no_grad():
#             if dynamic: 
#                 actual_state = nn.pad(state, actual_input_size)
#             else:
#                 actual_state = state
#             #Even though this looks like it would work with dynamic False I do not think it would. Should just assume it is dynamic for this branch tbh
#             action, _, _ = actor(actual_state, num_devices = tensor(state.shape[0]).unsqueeze(dim=0), batch_size = 1)
#             return action
    
#     def evaluate(state : Tensor) -> Tensor:
#         with torch.no_grad():
#             if dynamic: 
#                 actual_state = nn.pad(state, actual_input_size)
#             else:
#                 actual_state = state
#             return actor.evaluate(actual_state, tensor([state.shape[0]]))
    
#     return Agent(
#         target_update,
#         update_frequency,
#         learning_steps,
#         update,
#         call,
#         target_q.update,
#         actor.save,
#         evaluate
#     )
    
# def create_hdc_agent(input_size : int,
#                  output_size : int,
#                  hyper_dim : int,
#                  policy_lr : float,
#                  critic_lr : float,
#                  discount : float,
#                  tau : float,
#                  alpha_value : float,
#                  autotune : bool,
#                  target_update : int, #When the target should update
#                  update_frequency : int, #When the models should update,
#                  learning_steps : int, #Amount of gradient steps
#                  device : torch.device,
#                  dynamic : bool):
#     """Will create SAC agent based on HDC"""
    
#     actor_encoder = hdc.RBFEncoderFlatten(input_size, hyper_dim, dynamic)
#     critic_encoder = hdc.EXPEncoderFlatten(input_size, hyper_dim, dynamic)

#     target_q = hdc.TargetQFunction(tau, None)
#     alpha = hdc.Alpha(output_size, alpha_value, critic_lr, autotune=autotune)

#     actor = hdc.Actor(hyper_dim,
#                         output_size,
#                         policy_lr,
#                         actor_encoder,
#                         alpha,
#                         target_q,
#                         dynamic)
    
#     q_function = hdc.QFunction(hyper_dim,
#                                     output_size,
#                                     actor_encoder,
#                                     critic_encoder,
#                                     actor,
#                                     target_q,
#                                     alpha,
#                                     critic_lr,
#                                     discount,
#                                     dynamic)
    
#     target_q.set_actual(q_function)
    
#     for obj in [target_q, alpha, actor, q_function, actor_encoder, critic_encoder]:
#         obj.to(device)
    
#     def update(trans : Transition) -> Tensor:
#         """Will update following SAC equations for HDC and return losses in the form of a six dimension tensor in the form
#         Qfunc1, Qfunc2, Actor Loss, Entropy, Alpha Loss, Alpha Value"""
        
#         ce_state, q_info = q_function.update(trans)
#         actor_info = actor.update(trans, ce_state)
#         return torch.cat((q_info, actor_info))
        
    
#     def call(state : Tensor) -> Tensor:
#         """Will use HDC actor to determine action that should be taken at the given state"""
#         with torch.no_grad():
#             ae_state = actor_encoder(state.squeeze())
#             action, _, _ = actor(ae_state, num_devices = tensor(state.shape[0]).unsqueeze(dim=0), batch_size = 1)
#             return action
        
#     def evaluate(state : Tensor) -> Tensor:
#         with torch.no_grad():
#             ae_state = actor_encoder(state.flatten())
#             return actor.evaluate(ae_state)
        
#     return Agent(
#         target_update,
#         update_frequency,
#         learning_steps,
#         update,
#         call,
#         target_q.update,
#         actor.save,
#         evaluate
#     )
        
    

    
class Agent(ABC):
    
    def __init__(self,
                 target_update : int,
                 update_frequency : int,
                 learning_steps : int):
        
        self._learning_steps = learning_steps
        self._update_frequency = update_frequency
        self._target_update = target_update
    
    @abstractmethod
    def sample(self, state : Tensor | Data) -> Tensor:
        """Sample an action given a state

        Args:
            state (Tensor | Data): The state representation that will be sampled

        Returns:
            Tensor: The integer index of the action to be sampled
        """
        pass
    
    @abstractmethod
    def evaluate(self, state : Tensor | Data) -> Tensor:
        """Return the most likely action given a state

        Args:
            state (Tensor | Data): The state representation that will be sampled

        Returns:
            Tensor: The integer index of the action to be sampled
        """
        pass

    @abstractmethod
    def save(self, directory : str) -> None:
        """Will save the weights of the models into the given directory 

        Args:
            extension (str): The name of the directory
        """
        pass

    @abstractmethod  
    def add_data(self, trans : Transition) -> None:
        """Adds data from a given transition into memory

        Args:
            trans (Transition): The step transition to add to memory
        """
        pass
    
    @abstractmethod
    def param_update(self) -> dict[str : float]:
        """Will do a parameter update and return a dictionary of values that should be logged

        Returns:
            dict[str : float]: A dictionary mapping the name of the value and the value itself to be logged
        """
        pass

    @abstractmethod
    def target_param_update(self) -> None:
        """Will do the correct update for the target networks
        """
        pass
    
    def update(self, steps : int) -> None:
        """Will update the parameters of the models according to the current step in training

        Args:
            steps (int): The current step in training
        """
        if steps % self._update_frequency == 0:
            
            log_dicts = []
            
            for _ in range(self._learning_steps):
                log_dicts.append(self.param_update())
                
            self._log_data(log_dicts, steps)
            
        if steps % self._target_update == 0:
            self.target_param_update

    def calc_grad_norm(self, parameters : Iterable[Tensor]) -> float:
        """Will calculate the norm of the gradient across the parameters

        Args:
            parameters (Iterable[Tensor]): Parameters of a given network

        Returns:
            float: The norm of the parameters
        """
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    
    def polyak_average(self, actual_params : Iterable[Tensor], target_params : Iterable[Tensor], tau) -> None:
        """Will do a polyak average to update to update the target parameters given the actual parameters

        Args:
            actual_params (Iterable[Tensor]): Used to update
            target_params (Iterable[Tensor]): Will be updated
            tau (_type_): How much to update
        """
        for param, target_param in zip(actual_params, target_params):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _log_data(self, log_dicts: list[dict[str : float]], steps : int) -> None:
        new_log_dict = dict()
        number_dicts = len(log_dicts)

        for log_dict in log_dicts:
            for key, value in log_dict.items():
                new_log_dict[key] = value / number_dicts

        logger = LearningLogger()
        logger.log_scalars(new_log_dict, steps=steps)
    