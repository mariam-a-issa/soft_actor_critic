import csv
from pathlib import Path
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import git
import wandb as wb

from .config import Config


class LearningLogger:
    """Creates a logging class to handle specific types of logging for data about the perforamnce of the model"""
    _instance = None
    
    def __new__(cls, base_dir : str = None, 
                experiment_name : str = None, 
                hp_info : str = None, 
                config : Config = None):
        """Allows logger to follow singleton design"""
        if cls._instance is None or base_dir is not None or experiment_name is not None or hp_info is not None: #Build new instance when no new one exists or when the logging data is being changed
            cls._instance = super(LearningLogger, cls).__new__(cls)
            cls._instance._initialize(base_dir, experiment_name, hp_info, config)
        return cls._instance
    
    def _initialize(self, base_dir : str, experiment_name : str, hp_info : str, config : Config):
        """Will create tools used for logging. Done in _initialize instead of __init__ for the singleton to only do this when we are actually resetting in __new__"""
        self._loggers = dict()
        self._hparams = config.to_flat_dict() #Can be used at the end of a run alongside metrics to log hparams
        save_path = Path(base_dir) / experiment_name / hp_info / f'({config.seed})'
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        
        now = datetime.now()
        formatted = now.strftime("%m_%d_%H_%M")

        if config.tensorboard:
            tense_writer = SummaryWriter(save_path)
            self._loggers['tensorboard'] = tense_writer
        else:
            self._loggers['tensorboard'] = None
                
        if config.wandb:
            writer = wb.init(project=config.wandb_project_name, 
                             group=f'{formatted}_{sha[:config.num_sha_char]}_{experiment_name}_{hp_info}',
                             job_type = f'seed-{config.seed}',
                             config=self._hparams,
                             name = f'{formatted}_{sha[:config.num_sha_char]}_{experiment_name}_{hp_info}_seed-{config.seed}'
                             )
            self._loggers['wandb'] = writer
            writer.define_metric('Episode')
            writer.define_metric('Episodic Reward', step_metric='Episode')
        else:
            self._loggers['wandb'] = None

        if config.save_csv:
            _csv_of_hparams(save_path, self._hparams)
            
        self._cur_step = 0
            
    def add_x_axis_metric_labels(self, metrics_labels : dict[str : list[str]]) -> None:
        """Will create labels for specific metrics that are different from the standard 'step'
        
        metric_lables : Key is the added label (step metric) the list of strs is all of metrics that will have the label (step metric)
        
        """
        
        for key, value in metrics_labels.items():
            wb.define_metric(key)
            for metric in value:
                wb.define_metric(metric, step_metric=key)
            
    def log_scalars(self, data : dict[str, int|float], *, steps : int=None, episodes : int=None) -> None:
        """Will log data about each key in the dict as a scalar with its own plot"""
        
        if (steps is None and episodes is None) or (steps is not None and episodes is not None):
            raise TypeError("Needed to pass in steps or episodes. Cannot pass in steps and episodes")
        
        if self._loggers['tensorboard']:
            for key, value in data.items():
                reference = steps if steps is not None else episodes
                self._loggers['tensorboard'].add_scalar(key, value, reference)
                
        if self._loggers['wandb']:
            if episodes is not None:
                data['Episode'] = episodes
            self._loggers['wandb'].log(data, step=steps)
        
        if steps:
            self._cur_step = steps
            
    def cur_step(self)->int:
        """Will return roughly the current step"""
        return self._cur_step
    
    def close(self) -> None:
        """Will close the logger writing any additional information that is needed"""
        
        if self._loggers['tensorboard']:
            self._loggers['tensorboard'].close()
            
                
        if self._loggers['wandb']:
            self._loggers['wandb'].finish()

def _csv_of_hparams(log_dir : Path, h_params_dict : dict):
    """Creates a csv at the log dir with the given hyperparameters"""

    file = log_dir / 'hparams.csv'
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in h_params_dict.items():
            writer.writerow([key, value])
