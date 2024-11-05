from copy import copy
import os

from environment_run import train

MAIN_EXPERIMENT_NAME = 'nasim_hyperparam_alpha_medium_hdc'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'hdc_agent' : True,
    'alpha_value' : .25,
    'alpha_lr' : 5e-4,
    'critic_lr' : 5e-4,
    'hypervec_dim' : 4096,
    'hidden_size' : [512, 512],
    'policy_lr' : 5e-4,
    'sample_size' : 256,
    'tau' : 1,
    'seed' : None,
    'explore_steps' : 0,
    'buffer_size' : 10 ** 6,
    'learning_steps' : 1,
    'target_update' : 1000,
    'update_frequency' : 1,
    'environment_name' : 'nasim:Medium-v0',
    'max_steps' : 750000,
    'max_epi' : None,
    'eval_frequency' : 25,
    'num_evals' : 5,
    'autotune' : False,
    'wandb' : True,
    'tensorboard' : False
}

def train_hyper_param(name : list[str], values : list[float], seeds : list[int]):

    h_params = copy(OTHER_HPARAMS)

    for value in values:

        for n in name:
            h_params[n] = value

        for seed in seeds:
            
            h_params['seed'] = seed
            
            try:
                train(run_name = f'{name}({value})_seed({seed})', base_dir='runs', group_name = MAIN_EXPERIMENT_NAME, job_name = f'{name}_experiment', **h_params)
            except ValueError: 
                f = open(f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_seed({seed})/nan_v({value})_seed({seed}).txt', 'w', encoding='utf-8')
                f.write('I have NaNed')
                f.close()



if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #Needed since training will have to be deterministic. More info at https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    train_hyper_param(['alpha_value'], [.7], [3, 4])
