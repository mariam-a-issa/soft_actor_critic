from copy import copy

from train_loop import train

MAIN_EXPERIMENT_NAME = 'test'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'hdc_agent' : False,
    'alpha_scale' : .4,
    'alpha_lr' : 5e-4,
    'critic_lr' : 5e-4,
    'hypervec_dim' : 2048,
    'policy_lr' : 5e-4,
    'sample_size' : 512,
    'tau' : 1,
    'seed' : None,
    'explore_steps' : 10000,
    'buffer_size' : 10 ** 6,
    'learning_steps' : 4,
    'target_update' : 8000,
    'update_frequency' : 1
}

def train_hyper_param(name : str, values : list[float], seeds : list[int]):

    h_params = copy(OTHER_HPARAMS)

    for value in values:

        h_params[name] = value

        for seed in seeds:
            
            h_params['seed'] = seed
            
            try:
                train(f'{name}({value})_seed({seed})', log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment', **h_params)
            except ValueError: 
                f = open(f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_seed({seed})/nan_v({value})_seed({seed}).txt', 'w', encoding='utf-8')
                f.write('I have NaNed')
                f.close()



if __name__ == '__main__':
    train_hyper_param('tau', [1], [0,1,2,3,4])
