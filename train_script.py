from copy import copy
import os

from environment_run import train

MAIN_EXPERIMENT_NAME = 'nasim_tempcoef_hdc'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'hdc_agent' : True,
    'alpha_value' : .4,
    'alpha_lr' : 5e-4,
    'critic_lr' : 5e-4,
    'hypervec_dim' : 2048,
    'policy_lr' : 5e-4,
    'sample_size' : 256,
    'tau' : .005,
    'seed' : None,
    'explore_steps' : 0,
    'buffer_size' : 10 ** 6,
    'learning_steps' : 1,
    'target_update' : 1,
    'update_frequency' : 1,
    'environment_name' : 'nasim:Small-v0',
    'max_steps' : 150000,
    'eval_frequency' : 100,
    'num_evals' : 5,
    'autotune' : False
    
}

def train_hyper_param(name : str, values : list[float], seeds : list[int]):

    h_params = copy(OTHER_HPARAMS)

    for value in values:

        h_params[name] = value

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
    train_hyper_param('alpha_value', [.25, .5, .75, 1, 1.25, 1.5, 1.75, 2], [0, 1, 2, 3, 4])
