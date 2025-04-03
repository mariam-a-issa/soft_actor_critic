from copy import copy
import os

import torch

from utils import Config
from training_pipeline import train


MAIN_EXPERIMENT_NAME = 'nasimemu-medium-autotune-bundle-all-devices-and-permute-then-normalize-with-root-then-bundle-specific-device-h-param-sweep-critic-lr-correct-way'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'environment_info' : {'id' : 'NASimEmu-v0', 'emulate' : False, 'scenario_name' : '/home/ian/projects/hd_sac/NetworkAttackSimulator/nasim/scenarios/benchmark/medium.yaml', 'step_limit' : 100, 'augment_with_action' : True},
    'type_agent' : 'hdc_mil',
    'attention' : False,
    'graph' : False,
    'wandb' : True,
    'sample_size' : 64,
    'gpu_device' : 1,
    'policy_lr' : 3e-4,
    'critic_lr' : 3e-4
}

def train_hyper_param(name : str, values : list[float], seeds : list[int]):

    h_params = copy(OTHER_HPARAMS)

    for value in values:

        h_params[name] = value

        for seed in seeds:
            
            h_params['seed'] = seed

            try:
                train(run_name = f'{name}({value})_seed({seed})', base_dir='runs', group_name = MAIN_EXPERIMENT_NAME, job_name = f'{name}_experiment', config=Config().with_updates(**h_params))
            except ValueError as e: 
                directory_path = f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_seed({seed})/'
                os.makedirs(directory_path, exist_ok=True)
                f = open(f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_seed({seed})/nan_v({value})_seed({seed}).txt', 'w', encoding='utf-8')
                f.write('I have NaNed')
                f.close()


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #Needed since training will have to be deterministic. More info at https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    train_hyper_param('critic_lr', [3e-4], [0, 1, 2])
