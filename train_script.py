from train_loop import train

MAIN_EXPERIMENT_NAME = 'hdc_lr_explore_AS=.75'
HDC = True
NUM_RUNS = 3
OTHER_HPARAMS ={ #Just the default params that may be different than the ones in the training file
    'hdc_agent' : True,
    'alpha_scale' : .75
}

def train_script():
    """Will do multiple training loops"""
    
    lr = [.1, .08, .05, .03, .01, .005, .001]
    
    for value in lr:
        for i in range(NUM_RUNS):
            try:
                train(f'lr({value})_run({i})', policy_lr=value, critic_lr=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/lr_experiment', **OTHER_HPARAMS)
            except ValueError: 
                f = open(f'runs/{MAIN_EXPERIMENT_NAME}/lr_experiment/lr({value})_run({i})/nan_v({value})_run({i}).txt', 'w', encoding='utf-8')
                f.write('I have NaNed')
                f.close()
                
            
    #actor_lr = [.1, .05, .01, .005, .001, .0005, .0001]

    #for value in actor_lr:
    #    for i in range(NUM_RUNS):
    #        train(f'policy_lr({value})_run({i})', policy_lr=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/policy_lr_experiment', **OTHER_HPARAMS)

    #critic_lr = [.1, .05, .01, .005, .001, .0005, .0001]

    #for value in critic_lr:
    #    for i in range(NUM_RUNS):
    #        train(f'critic_lr({value})_run({i})', critic_lr=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/critic_lr_experiment',  **OTHER_HPARAMS)
    
    #alpha_lr = []

    #for value in alpha_lr:
    #    train(f"alpha_lr'{value}'", alpha_lr=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/alpha_lr_experiment', hdc_agent=HDC, **OTHER_HPARAMS)

    #tau = [.1, .05, .01, .005, .001, .0005, .0001]
      
    #for value in tau:
    #    train(f"tau'{value}'", tau=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/tau_experiment', hdc_agent=HDC)
        
    #hyper_dim = [256, 512, 1024, 2048, 4096, 8192]
    
    #for value in hyper_dim:
    #    for i in range(NUM_RUNS):
    #        train(f'hyper_dim({value})_run({i})', hypervec_dim=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/hyper_dim_experiment', **OTHER_HPARAMS)
        
    #alpha_scale = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65] # .7, .75, .8, .85, .9, .95, 1]

    #for value in alpha_scale:
    #    for i in range(NUM_RUNS):
    #        train(f"alpha_scale'{value}_run{i}'", alpha_scale=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/alpha_scale_experiment', hdc_agent=HDC)

    #sample_size = [32, 64, 128, 256, 512]

    #for value in sample_size:
    #    for i in range(NUM_RUNS):
    #        train(f'sample_size({value})_run({i})', buffer_size=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/sample_experiment', **OTHER_HPARAMS)

if __name__ == '__main__':
    train_script()