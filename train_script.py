from train_loop import train

def train_script():
    """Will do multiple training loops"""

    actor_lr = [.01, .005, .001, .005, .0001, .00005, .00001]

    for value in actor_lr:
        train(f"policy_lr'{value}'", policy_lr=value, log_dir='runs/policy_lr_experiment')

    critic_lr = [.01, .005, .001, .0005, .0001, .00005, .00001]

    for value in critic_lr:
        train(f"critic_lr'{value}'", critic_lr=value, log_dir='runs/critic_lr_experiment')

    alpha_lr = [.01, .005, .001, .0005, .0001, .00005, .00001]

    for value in alpha_lr:
        train(f"alpha_lr'{value}'", alpha_lr=value, log_dir='runs/alpha_lr_experiemnt')

    tau = [.1, .05, .01, .005, .001, .0005, .0001]
      
    for value in tau:
        train(f"tau'{value}'", tau=value, log_dir='runs/tau_experiment')

    alpha_scale = [.60, .65, .7, .75, .8, .85, .9, .95, 1.0, 1.05, 1.1]

    for value in alpha_scale:
        train(f"alpha_scale'{value}'", alpha_scale=value, log_dir='runs/alpha_scale_experiment')

    buffer_size = [32, 64, 128, 512]

    for value in buffer_size:
        train(f"buffer_size'{value}'", buffer_size=value, log_dir='runs/buffer_size_experiment')

if __name__ == '__main__':
    train_script()