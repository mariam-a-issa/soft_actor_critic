from copy import copy
import os

from environment_run import train

MAIN_EXPERIMENT_NAME = 'nasimemu-mil-alpha-stp100-aug-no-clean-id--seperate-encoders-autotune-alpha-attention'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'hdc_agent' : False,
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
    'sample_size' : 64,
    'tau' : .005,
    'target_update' : 1,
    'seed' : None,
    'explore_steps' : 0,
    'buffer_size' : 10 ** 6,
    'learning_steps' : 1,
    'update_frequency' : 1,
    'environment_info' : {'id' : 'NASimEmu-v0', 'emulate' : False, 'scenario_name' : '/home/mariamai/projects/hd_sac/NASimEmu/scenarios/corp.v2.yaml', 'step_limit' : 100, 'augment_with_action' : True},
    'max_steps' : 250000,
    'eval_frequency' : 10,
    'num_evals' : 5,
    'tensorboard' : False,
    'wandb' : False,
    'dynamic' : True,
    'target_start' : .8,
    'target_end' : .2,
    'midpoint' : .4,
    'slope' : 7, 
    'attention' : True,
    'num_heads' : 2
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
    train_hyper_param('target_start', [.8], [0, 1, 2])
