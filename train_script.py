from copy import copy

from train_loop import train

MAIN_EXPERIMENT_NAME = 'test'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'hdc_agent' : False,
    'alpha_scale' : .2,
    'alpha_lr' : 3e-4,
    'critic_lr' : 3e-4,
    'hypervec_dim' : 2048,
    'policy_lr' : 3e-4,
    'sample_size' : 256,
    'tau' : .005,
    'seed' : 0
}

def train_hyper_param(name : str, values : list[float]):

    h_params = copy(OTHER_HPARAMS)

    for value in values:

        h_params[name] = value

        for i in range(NUM_RUNS):
            try:
                train(f'{name}({value})_run({i})', log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment', **h_params)
            except ValueError: 
                f = open(f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_run({i})/nan_v({value})_run({i}).txt', 'w', encoding='utf-8')
                f.write('I have NaNed')
                f.close()



if __name__ == '__main__':
    train_hyper_param('seed', [0, 1, 2])
