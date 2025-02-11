import random
import csv
from copy import deepcopy
from pathlib import Path

import torch
from torch import Tensor
import numpy as np
from nasimemu import env_utils
from numpy.typing import NDArray

#import gymnasium
import gym

from discrete import create_hdc_agent, create_nn_agent, create_mil_nn_agent
from utils import MemoryBuffer, Transition, LearningLogger
from .evaluate import evaluate
from .helpers import convert_int_action, clean_state

#Hyperparameters

LR = 3e-4
HIDDEN_LAYER_SIZE = 256 #They use 64
POS_ENC_SIZE = 8
HYPER_VEC_DIM = 2048
POLICY_LR = LR
CRITIC_LR = LR
ALPHA_LR = LR
AUTOTUNE = True
DISCOUNT = .99
TAU = .005
ALPHA_VALUE = .75 #Kinda for the lunar lander
TARGET_UPDATE = 1 
UPDATE_FREQUENCY = 1 
LEARNING_STEPS = 4
EXPLORE_STEPS = 0
BUFFER_SIZE = 10 ** 6
SAMPLE_SIZE = 256
EVAL_FREQUENCY = 15
NUM_EVALS = 3

LOG_DIR = './runs/large__alpha'

MAX_STEPS = 6e5


MAGIC_CORP_NUM = 20 #Magic number involved with indexing in the corp scenario

def train(
        run_name : str = '',  #Multiple runs inside of a job (Usually for different seeds)
        base_dir : str = LOG_DIR, #Root of all experiments
        group_name : str = '', #Groups of various experiments
        job_name : str = '',   #Individual jobs in the experiment
        hidden_size : int = HIDDEN_LAYER_SIZE,
        pos_enc_size : int = POS_ENC_SIZE,
        policy_lr : float = POLICY_LR,
        critic_lr : float = CRITIC_LR,
        alpha_lr : float = ALPHA_LR,
        discount : float = DISCOUNT,
        tau : float = TAU,
        alpha_value :float = ALPHA_VALUE,
        autotune : bool = AUTOTUNE,#Can either be target scaling or actual temperature coefficient
        target_update : int = TARGET_UPDATE,
        update_frequency : int = UPDATE_FREQUENCY,
        explore_steps : int = EXPLORE_STEPS ,
        buffer_size : int = BUFFER_SIZE,
        sample_size : int = SAMPLE_SIZE,
        max_steps : int = MAX_STEPS,
        hdc_agent : bool = False,
        mil_agent : bool = True,
        hypervec_dim : int = HYPER_VEC_DIM,
        environment_info : dict = {'id' : 'LunarLander-v2'}, #Currently using gym as nasimemu uses gym
        seed : int = None,
        gpu : bool = True,
        learning_steps : int = LEARNING_STEPS,
        eval_frequency : int = EVAL_FREQUENCY,
        num_evals : int = NUM_EVALS,
        tensorboard : bool = True,
        wandb : bool = True,
        dynamic : bool = False,
        grad_clip : float = None,
        target_start : float = .8,
        target_end : float = .2,
        midpoint : float = .5,
        slope : float = 5,
        attention : bool = True,
        num_heads : int = 2) -> None:
    """Will be the main training loop"""
    
    main_dir = Path(base_dir)
    run_path = main_dir / group_name / job_name / run_name
    
    h_params_dict = deepcopy(locals())
    del h_params_dict['run_name']
    del h_params_dict['base_dir']
    del h_params_dict['group_name']
    del h_params_dict['job_name']
    del h_params_dict['environment_info']
    
    for key, value in environment_info.items():
        h_params_dict[key] = value
        
    _csv_of_hparams(run_path, h_params_dict)

    buffer = MemoryBuffer(buffer_size, sample_size, random, dynamic)

    logger = LearningLogger(base_dir, group_name, job_name, run_name, h_params_dict, tensorboard=tensorboard, wandb=wandb)
    
    #"LunarLander-v2"
    #"CartPole-v1"
    #"MountainCar-v0"
    env = gym.make(**environment_info)
    env.reset()
    
    if environment_info['id'] == 'NASimEmu-v0':
        #Got from their config file on how to get sorta of an idea of the size of state and action spaces
        action_space = len(env.action_list)
        s = env.reset()
        state_space = s.shape[1] #- MAGIC_CORP_NUM TODO if going to clean put this back in
        dynamic = True
    else:
        action_space = env.action_space.n
        state_space = env.observation_space.shape[0]
        dynamic = False
    
    if seed is not None:
        torch.manual_seed(seed) 
        torch.use_deterministic_algorithms(True)
        random.seed(seed)
        np.random.seed(seed)

    if torch.cuda.is_available() and gpu:
        device = f'cuda:{torch.cuda.current_device()}'
    else:
       device = 'cpu'

    device_obj = torch.device(device)

    torch.set_default_device(device_obj)

    if mil_agent:
        agent = create_mil_nn_agent(
            state_space,
            action_space,
            hidden_size, 
            pos_enc_size,
            policy_lr,
            critic_lr,
            alpha_lr,
            discount,
            tau,
            alpha_value,
            target_update,
            update_frequency,
            learning_steps,
            device_obj,
            buffer_size,
            sample_size,
            grad_clip,
            target_start,
            target_end,
            slope,
            midpoint,
            max_steps,
            autotune,
            random,
            attention,
            num_heads  
        )
    else:
        if hdc_agent:
            agent = create_hdc_agent(
                state_space,
                action_space,
                hypervec_dim,
                policy_lr,
                critic_lr,
                discount,
                tau,
                alpha_value,
                autotune,
                target_update,
                update_frequency,
                learning_steps,
                device_obj,
                dynamic  
            )
        else:
            agent = create_nn_agent(
                state_space,
                action_space,
                hidden_size,
                policy_lr,
                critic_lr,
                alpha_lr,
                discount,
                tau,
                alpha_value,
                autotune,
                target_update,
                update_frequency,
                learning_steps,
                device_obj,
                dynamic,
                grad_clip
            )

    steps = 0
    num_epi = 0
    epi_reward = 0
    
    def get_action(state : Tensor, real_state : NDArray) -> tuple[tuple[tuple[int, int], int], Tensor]:
        """Will get the action depending on exploring or doing the current policy
           Will return the NASimEmu action and the integer action as a Tensor"""
        if explore_steps <= steps:
            action = agent(state) 
            return convert_int_action(action.data, env, real_state), action
        else:
            action = random.randint(0, env.action_space.n-1) #Fix so that it takes into account padded actions depending on size of state
            return convert_int_action(action, env ,state), torch.tensor(action) 
    
    real_state = env.reset()
    state = torch.tensor(clean_state(real_state), device=device_obj, dtype=torch.float32)

    try:
        while max_steps > steps:
            action_nas, action = get_action(state, real_state)
            next_state, reward, done, _ = env.step(action_nas)
            real_state = next_state
            next_state = clean_state(next_state)
            next_state = torch.tensor(next_state, device=device_obj, dtype=torch.float32).view(-1, state_space)
            trans = Transition( #states will be tensors, actions will be tensor integers, the reward will be a float, and terminated will be a bool
                state=state,
                action=action,
                next_state=next_state,
                reward=torch.tensor([reward], device=device_obj, dtype=torch.float32),
                done=torch.tensor([False], device=device_obj, dtype=torch.float32) #Currently the agent never actually comes to a point where it makes a move that terminates. Therefor done should not be incorporated 
            )
            
            epi_reward += reward

            agent.add_data(trans)
            
            if explore_steps <= steps:
                agent.update(steps)

            steps += 1

            if done:
                
                LearningLogger().log_scalars({'Training reward' : epi_reward}, episodes=num_epi)
                
                epi_reward = 0
                if explore_steps <= steps:
                    num_epi += 1
                    if num_epi % eval_frequency == 0:
                        evaluate(deepcopy(env), agent, num_evals, num_epi) #Need to deepcopy so that we keep the environment the same when training or else the state the environment will be in will be different from the state that is in next_state

            state = next_state
    except Exception as e:
        print(e)
        raise e
    
    finally:
        agent.save_actor(job_name)
        env.close()
        logger.close()

def _csv_of_hparams(log_dir : Path, h_params_dict : dict):
    """Creates a csv at the log dir with the given hyperparameters"""

    file = log_dir / 'hparams.csv'
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in h_params_dict.items():
            writer.writerow([key, value])
        #writer.writerow(['clip_critic', True])

if __name__ == '__main__':
    for i in range(3):
        train(i, hdc_agent=False)
