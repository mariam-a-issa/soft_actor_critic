from copy import copy

from train_loop import train

MAIN_EXPERIMENT_NAME = 'nasim_alpha_hdc_large'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'hdc_agent' : True,
    'alpha_scale' : .4,
    'alpha_lr' : 3e-4,
    'critic_lr' : 3e-4,
    'hypervec_dim' : 2048,
    'policy_lr' : 3e-4,
    'sample_size' : 64,
    'tau' : .003,
    'seed' : None,
    'explore_steps' : 0,
    'buffer_size' : 10 ** 6,
    'learning_steps' : 1,
    'target_update' : 1,
    'update_frequency' : 1,
    'environment_name' : 'nasim:MediumMultiSite-v0',
    'max_steps' : 20000
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
    train_hyper_param('alpha_scale', [.1, .15, .2, .25, .3, .35, .4, .5, .6], [0,1,2,3,4])
