from copy import copy
import os

from train_loop import train

MAIN_EXPERIMENT_NAME = 'test'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'hdc_agent' : True,
    'alpha_scale' : .4,
    'alpha_lr' : 3e-4,
    'critic_lr' : 3e-4,
    'hypervec_dim' : 2048,
    'policy_lr' : 3e-4,
    'sample_size' : 256,
    'tau' : .005,
    'seed' : None,
    'explore_steps' : 0,
    'buffer_size' : 10 ** 6,
    'learning_steps' : 1,
    'target_update' : 1,
    'update_frequency' : 1,
    'environment_name' : 'nasim:Tiny-v0',
    'max_steps' : 20000
}

def train_hyper_param(name : str, values : list[float], seeds : list[int]):

    h_params = copy(OTHER_HPARAMS)

    for value in values:

        h_params[name] = value

        for seed in seeds:
            
            h_params['seed'] = seed
            
            try:
                train(f'{name}({value})_seed({seed})', log_dir=f'runs/{MAIN_EXPERIMENT_NAME}', experiment_name = f'{name}_experiment', **h_params)
            except ValueError: 
                f = open(f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_seed({seed})/nan_v({value})_seed({seed}).txt', 'w', encoding='utf-8')
                f.write('I have NaNed')
                f.close()



if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #Needed since training will have to be deterministic. More info at https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    train_hyper_param('alpha_scale', [.1], [0])
