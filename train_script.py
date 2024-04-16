from train_loop import train

def train_script():
    """Will do multiple training loops"""
    
    actor_lr = [.01, .005, .001, .0005, .0001, .00005, .00001]

    for value in actor_lr:
        train(f"policy_lr'{value}'", policy_lr=value, log_dir='runs/policy_lr_experiment')

if __name__ == '__main__':
    train_script()