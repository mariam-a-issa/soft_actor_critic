from train_loop import train

def train_script():

    alpha_scale = .7
    critic_lr = .0001
    tau = .05

    for i in range(3):
        train(f'experiment{i}', 
              alpha_scale=alpha_scale, 
              critic_lr=critic_lr,
              tau=tau)