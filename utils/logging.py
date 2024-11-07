from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import wandb
import csv

PROJECT_NAME = 'NASIM Hyperparamtune'

class LearningLogger:
    """Creates a logging class to handle specific types of logging for data about the perforamnce of the model"""
    _instance = None
    
    def __new__(cls, base_dir : str = None, group_name : str = None, job_name : str = None, run_name : str = None, hparams_config : dict = None, tensorboard : bool = True, wand : bool = True):
        """Allows logger to follow singleton design"""
        if cls._instance is None or base_dir is not None or group_name is not None or job_name is not None or run_name is not None: #Build new instance when no new one exists or when the logging data is being changed
            cls._instance = super(LearningLogger, cls).__new__(cls)
            cls._instance._initialize(base_dir, group_name, job_name, run_name, hparams_config, tensorboard, wand)
        return cls._instance
    
    def _initialize(self, base_dir : str, group_name : str, job_name : str, run_name : str, hparams_config : dict, tensorboard : bool = True, wand : bool = True):
        """Will create tools used for logging. Done in _initialize instead of __init__ for the singleton to only do this when we are actually resetting in __new__"""
        self._loggers = dict()
        self._hparams = hparams_config #Can be used at the end of a run alongside metrics to log hparams
        
        if tensorboard:
            save_dir = Path(base_dir) / group_name / job_name / run_name
            tense_writer = SummaryWriter(save_dir)
            self._loggers['tensorboard'] = tense_writer
            _csv_of_hparams(save_dir, hparams_config)
        else:
            self._loggers['tensorboard'] = None
                
        if wand:
            writer = wandb.init(project=PROJECT_NAME, name=group_name + '/' + job_name + '/' + run_name, config=hparams_config)
            self._loggers['wandb'] = writer
            writer.define_metric('Episode')
            writer.define_metric('Episodic Reward', step_metric='Episode')
            
            
        else:
            self._loggers['wandb'] = None
            
    def add_x_axis_metric_labels(self, metrics_labels : dict[str : list[str]]) -> None:
        """Will create labels for specific metrics that are different from the standard 'step'
        
        metric_lables : Key is the added label (step metric) the list of strs is all of metrics that will have the label (step metric)
        
        """
        
        for key, value in metrics_labels.items():
            wandb.define_metric(key)
            for metric in value:
                wandb.define_metric(metric, step_metric=key)
            
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