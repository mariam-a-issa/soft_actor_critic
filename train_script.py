from copy import copy
import os
from pathlib import Path
import argparse

from git import Repo

from utils import Config
from training_pipeline import train

PROJECT_NAME = 'New Encoder'
MAIN_EXPERIMENT_NAME = 'test'
NUM_RUNS = 1
OTHER_HPARAMS = { #Just the default params that may be different than the ones in the training file
    'wandb_project_name' : PROJECT_NAME,
    'environment_info' : {'id' : 'NASimEmu-v0', 'emulate' : False, 'scenario_name' : '/home/ian/projects/hd_sac/NetworkAttackSimulator/nasim/scenarios/benchmark/medium.yaml', 'step_limit' : 100, 'augment_with_action' : True},
    'type_agent' : 'nn_mil',
    'wandb' : False,
    'tensorboard' : False
}

def train_hyper_param(name : str, values : list[float], seeds : list[int]):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    if not args.debug:
        if not _check_git_clean():
            raise RuntimeError("Commit latest changes before running an experiment")
        
        note = _get_note()
    else:
        note = None
        
    h_params = copy(OTHER_HPARAMS)
    h_params['notes'] = note
    for value in values:

        h_params[name] = value

        for seed in seeds:
            
            h_params['seed'] = seed

            try:
                train(base_dir='runs', experiment_name = MAIN_EXPERIMENT_NAME, hp_info = f'{name}_{value}', config=Config().with_updates(**h_params))
            except ValueError as e: 
                directory_path = f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_seed({seed})/'
                os.makedirs(directory_path, exist_ok=True)
                f = open(f'runs/{MAIN_EXPERIMENT_NAME}/{name}_experiment/{name}({value})_seed({seed})/nan_v({value})_seed({seed}).txt', 'w', encoding='utf-8')
                f.write('I have NaNed')
                f.close()
                raise e

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
    train_hyper_param('policy_lr', [3e-4], [0, 1, 2])
