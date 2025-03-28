from copy import copy
import os

import torch

from utils import Config
from environment_run import train


MAIN_EXPERIMENT_NAME = 'nasimemu-medium-bind-encoding-standard-hparams-hdc'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'hdc_agent' : True,
    'mil_agent' : True,
    'alpha_value' : .4, #The tempurature coefficient or scaling factor for the target entropy when autotuning 
    'autotune' : True,
    'alpha_lr' : 3e-4,
    'critic_lr' : 3e-4,
    'policy_lr' : 3e-4,
    'hypervec_dim' : 4096,
    'hidden_size' : 128,
    'pos_enc_size' : 8,
    'grad_clip' : 5,
    'sample_size' : 512,
    'tau' : .005,
    'target_update' : 1,
    'seed' : None,
    'explore_steps' : 0,
    'buffer_size' : 10 ** 6,
    'learning_steps' : 1,
    'update_frequency' : 1,
    'environment_info' : {'id' : 'NASimEmu-v0', 'emulate' : False, 'scenario_name' : '/home/ian/projects/hd_sac/NetworkAttackSimulator/nasim/scenarios/benchmark/medium.yaml', 'step_limit' : 100, 'augment_with_action' : True},
    'max_steps' : 250000,
    'eval_frequency' : 10,
    'num_evals' : 5,
    'tensorboard' : False,
    'save_csv' : False,
    'wandb' : True,
    'dynamic' : True,
    'target_start' : .8,
    'target_end' : .2,
    'midpoint' : .45,
    'slope' : 6, 
    'attention' : False,
    'num_heads' : 2,
    'graph' : False,
    'messages_passed' : 2,
    'gpu' : True
}

def train_hyper_param(name : str, values : list[float], seeds : list[int]):

    h_params = copy(OTHER_HPARAMS)

    for value in values:

        h_params[name] = value

        for seed in seeds:
            
            h_params['seed'] = seed
            
            torch.cuda.memory._record_memory_history(
                max_entries=10000
            )

            try:
                train(run_name = f'{name}({value})_seed({seed})', base_dir='runs', group_name = MAIN_EXPERIMENT_NAME, job_name = f'{name}_experiment', config=Config().with_updates(**h_params))
            except ValueError as e: 
                f = open(f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_seed({seed})/nan_v({value})_seed({seed}).txt', 'w', encoding='utf-8')
                f.write('I have NaNed')
                f.close()
            except torch.OutOfMemoryError:
                try:
                    torch.cuda.memory._dump_snapshot(f"{'test'}.pickle")
                except Exception as e:
                    print(f"Failed to capture memory snapshot {e}")
                torch.cuda.memory._record_memory_history(enabled=None)



if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #Needed since training will have to be deterministic. More info at https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    train_hyper_param('critic_lr', [3e-4], [0, 1, 2])
