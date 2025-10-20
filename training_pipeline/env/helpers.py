import random

import torch
import numpy as np

from utils import Config


def setup_env(config : Config) -> torch.device:
    """Will setup the computer training environment

    Args:
        config (Config): The config object of the current training run

    Returns:
        torch.device: The device object that the models will be running on
    """

    if config.seed is not None:
        torch.manual_seed(config.seed) 
        torch.use_deterministic_algorithms(True, warn_only=True)
        random.seed(config.seed)
        np.random.seed(config.seed)

    if torch.cuda.is_available() and config.gpu:
        device = f'cuda:{config.gpu_device}'
    else:
       device = 'cpu'

    device_obj = torch.device(device)

    torch.set_default_device(device_obj)

    return device_obj
