from copy import copy
import os
import argparse
from utils import Config
from training_pipeline import train
from git import Repo
from pathlib import Path

OTHER_HPARAMS = {
    'environment_info' : {'id' : 'nasim:TinyPO-v0', 'flat_actions' : True, 'flat_obs' : True},
    'type_agent' : 'nn',
    'wandb' : False,
    'tensorboard' : False,
    'g_drive' : False,
    'max_steps' : 20000,
}

def train_hyper_param():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('-j', '--json_path', type=str, help='File Path to JSON File', default=None)
    parser.add_argument('-n', '--experiment_name', type=str, help='Name of the current experiment', default=None)
    parser.add_argument('-hp', '--hyperparam', type=str, help='Name of the hyperparameter to tune', default=None)
    parser.add_argument('-v', '--values', type=str, help='Comma seperated list of hyperparam values to test', default=None)
    parser.add_argument('-s', '--seeds', type=str, help='Comma seperated list of seeds', default=None)
    parser.add_argument('-p', '--project_name', type=str, help='Name of the wandb project', default=None)
    args = parser.parse_args()

    if not args.debug:
        if not _check_git_clean():
            raise RuntimeError("Commit latest changes before running an experiment")
        note = _get_note()
    else:
        train(config=Config().with_updates(**OTHER_HPARAMS))
        return

    if args.json_path:
        h_params = {}
        config = Config().update_from_json(args.json_path)
    else:
        h_params = copy(OTHER_HPARAMS)
        config = Config()

    if not args.experiment_name or not args.seeds or not args.project_name:
        raise TypeError('Need to provide: experiment name, project name, and seeds for training')
    
    experiment_name = args.experiment_name
    seeds = args.seeds.split(',')
    h_params['wandb_project_name'] = args.project_name
    
    if args.hyperparam:
        name = args.hyperparam
        values = args.values.split(',')
    else:
        values = [None]

    for value in values:
        if value:
            h_params[name] = value

        if value:
            hp_info = f'{name}_{value}'
        else:
            hp_info = ''

        for seed in seeds:
            h_params['seed'] = int(seed)

            train(experiment_name=experiment_name, hp_info=hp_info, config=config.with_updates(**h_params))

if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #Needed since training will have to be deterministic. More info at https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    train_hyper_param()
