from copy import copy
import os
from pathlib import Path
import argparse

from git import Repo

from utils import Config
from training_pipeline import train

OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'environment_info' : {'id' : 'nasim:TinyPO-v0', 'flat_actions' : True, 'flat_obs' : True},
    'type_agent' : 'nn',
    'wandb' : False,
    'tensorboard' : False,
    'g_drive' : True,
    'max_steps' : 200
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

    if args.json_path:
        h_params = {}
        config = Config().update_from_json(args.json_path)
    else:
        h_params = copy(OTHER_HPARAMS)
        config = Config()

    if not args.experiment_name or not args.seeds or not args.project_name:
        raise TypeError('Need to provide experiemnt and project name as well as seeds when training')
    
    experiment_name = args.experiment_name
    seeds = args.seeds.split(',')
    h_params['wandb_project_name'] = args.project_name
    
    if args.hyperparam:
        name = args.hyperparam
        values = args.values.split(',')
    else:
        values = [None]

    h_params['notes'] = note

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

def _check_git_clean(repo_path='.'):
    """
    Checks if the Git repository at repo_path is clean (no unstaged or uncommitted changes).
    Returns True if clean, False otherwise.
    """
    repo = Repo(repo_path)
    return not repo.is_dirty(untracked_files=True)

def _get_note() -> str:
    try:
        file_path = Path('./note.txt')
        with open(file_path, 'r') as file:
            content = file.read()
            print('Notes:\n')
            print(content)
            os.remove(file_path)
            return content 
    except FileNotFoundError:
        raise RuntimeError(f"No note file was created. Make sure to have notes in the note.txt file before running an experiment")

if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' #Needed since training will have to be deterministic. More info at https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    train_hyper_param()
