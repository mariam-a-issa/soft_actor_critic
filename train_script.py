from train_loop import train

MAIN_EXPERIMENT_NAME = 'first_hdc_implementation'
HDC = True

def train_script():
    """Will do multiple training loops"""

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
        train(f"hyper_dim'{value}'", hypervec_dim=hyper_dim, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/hyper_dim_experiment', hdc_agent=HDC)

    alpha_scale = [.60, .65, .7, .75, .8, .85, .9, .95, 1.0, 1.05, 1.1]

    for value in alpha_scale:
        train(f"alpha_scale'{value}'", alpha_scale=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/alpha_scale_experiment', hdc_agent=HDC)

    buffer_size = [32, 64, 128, 512]

    for value in buffer_size:
        train(f"buffer_size'{value}'", buffer_size=value, log_dir=f'runs/{MAIN_EXPERIMENT_NAME}/buffer_size_experiment', hdc_agent=HDC)

if __name__ == '__main__':
    train_script()