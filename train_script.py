from train_loop import train

MAIN_EXPERIMENT_NAME = 'alpha_scale_hyper_param_tune'
HDC = False
NUM_RUNS = 3

def train_script():
    """Will do multiple training loops"""

    '''
    actor_lr = [.01, .005, .001, .0005, .0001, .00005, .00001]

    for value in actor_lr:
        train(f"policy_lr'{value}'", policy_lr=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/policy_lr_experiment', hdc_agent=HDC)

    critic_lr = [.01, .005, .001, .0005, .0001, .00005, .00001]

    for value in critic_lr:
        train(f"critic_lr'{value}'", critic_lr=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/critic_lr_experiment', hdc_agent=HDC)
    
    alpha_lr = [.01, .005, .001, .0005, .0001, .00005, .00001]

    for value in alpha_lr:
        train(f"alpha_lr'{value}'", alpha_lr=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/alpha_lr_experiment', hdc_agent=HDC)

    tau = [.1, .05, .01, .005, .001, .0005, .0001]
      
    for value in tau:
        train(f"tau'{value}'", tau=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/tau_experiment', hdc_agent=HDC)
        
    hyper_dim = [256, 512, 1024, 2048, 4096, 8192]
    
    for value in hyper_dim:
        train(f"hyper_dim'{value}'", hypervec_dim=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/hyper_dim_experiment', hdc_agent=HDC)
    '''
        
    alpha_scale = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1]

    for value in alpha_scale:
        for i in range(NUM_RUNS):
            train(f"alpha_scale'{value}_run{i}'", alpha_scale=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/alpha_scale_experiment', hdc_agent=HDC)

    '''
    sample_size = [32, 64, 128, 256, 512]

    for value in sample_size:
        train(f"sample_size'{value}'", buffer_size=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/sample_experiment', hdc_agent=HDC)

def extra_train():
    sample_size = [32, 64, 128, 256, 512]

    for value in sample_size:
        train(f"sample_size'{value}'", sample_size=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/sample_experiment', hdc_agent=True)
        
def average_train():
    hdc_dict = {
        'alpha_lr' : 1e-5,
        'alpha_scale' : .6,
        'critic_lr' : .005,
        'hypervec_dim' : 2048,
        'policy_lr' : 1e-5,
        'sample_size' : 512,
        'tau' : .03
    }
    
    nn_dict = {
        'alpha_lr' : 1e-5,
        'alpha_scale' : .7,
        'critic_lr' : .01,
        'policy_lr' : .01,
        'tau' : .005
    }
    
    #for i in range(3):
    #   train(str(i), log_dir='runs/fixed_g_hparam/nn', **nn_dict, hdc_agent=False)
        
    for i in range(3):
        train(str(i), log_dir='runs/fixed_g_hparam/hdc', **hdc_dict, hdc_agent=True)
'''

if __name__ == '__main__':
    train_script()