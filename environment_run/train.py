import random
import csv
from copy import deepcopy
from pathlib import Path
import socket

import torch
from torch import Tensor
import numpy as np

import gymnasium as gym

from discrete import create_hdc_agent, create_nn_agent
from utils import MemoryBuffer, Transition, LearningLogger
from .evaluate import evaluate

#Hyperparameters

LR = 3e-4
HIDDEN_LAYER_SIZE = [512, 512]
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
MAX_EPI = 3e5

def train(
        run_name : str = '',  #Multiple runs inside of a job (Usually for different seeds)
        base_dir : str = LOG_DIR, #Root of all experiments
        group_name : str = '', #Groups of various experiments
        job_name : str = '',   #Individual jobs in the experiment
        hidden_size : list[int] = HIDDEN_LAYER_SIZE,
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
        max_epi : int = MAX_EPI,
        hdc_agent : bool = False,
        hypervec_dim : int = HYPER_VEC_DIM,
        environment_name : str = 'LunarLander-v2',
        seed : int = None,
        gpu : bool = True,
        learning_steps : int = LEARNING_STEPS,
        eval_frequency : int = EVAL_FREQUENCY,
        num_evals : int = NUM_EVALS,
        wandb : bool = True,
        tensorboard : bool = True) -> None:
    """Will be the main training loop"""
    
    main_dir = Path(base_dir)
    run_path = main_dir / group_name / job_name / run_name
    
    h_params_dict = deepcopy(locals())
    del h_params_dict['run_name']
    del h_params_dict['base_dir']
    del h_params_dict['group_name']
    del h_params_dict['job_name']
    h_params_dict['host_name'] = socket.gethostname()
    _csv_of_hparams(run_path, h_params_dict)

    buffer = MemoryBuffer(buffer_size, sample_size, random)

    logger = LearningLogger(base_dir, group_name, job_name, run_name, h_params_dict, wand=wandb, tensorboard=tensorboard)
    
    #"LunarLander-v2"
    #"CartPole-v1"
    #"MountainCar-v0"
    env = gym.make(environment_name)
    
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

    
    if hdc_agent:
        agent = create_hdc_agent(
            env.observation_space.shape[0],
            env.action_space.n,
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
            device_obj   
        )
    else:
        agent = create_nn_agent(
            env.observation_space.shape[0],
            env.action_space.n,
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
            device_obj
        )

    steps = 0
    num_epi = 0
    
    def get_action(s : Tensor) -> Tensor:
        if explore_steps <= steps:
            return agent(s)
        else:
            return torch.tensor(random.randint(0, env.action_space.n-1))
        
    state = torch.tensor(env.reset(seed=seed)[0], device=device_obj, dtype=torch.float32)

    try:
        while (max_steps is not None and max_steps > steps) or (max_epi is not None and max_epi > num_epi):
            action = get_action(state).unsqueeze(dim = 0)
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated
            next_state = torch.tensor(next_state, device=device_obj, dtype=torch.float32)
            trans = Transition( #states will be np arrays, actions will be tensors, the reward will be a float, and terminated will be a bool
                state,
                action,
                next_state,
                torch.tensor([reward], device=device_obj, dtype=torch.float32),
                torch.tensor([terminated], device=device_obj, dtype=torch.float32)
            )

            buffer.add_data(trans)
            
            if explore_steps <= steps:
                agent.update(buffer, steps)

            steps += 1

            if done:
                next_state = torch.tensor(env.reset()[0], device=device_obj, dtype=torch.float32)
                num_epi += 1
                if num_epi % eval_frequency == 0:
                    evaluate(env, agent, num_evals, num_epi)
                    
                
                
            state = next_state

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
