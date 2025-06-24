from .mil_hdc import MILHDCAgent
from .mil_nn import MILNNAgent
from .nn import MLPNNAgent
from .agents import Agent
from utils import Config


def create_agent(state_space_size: int, action_space_size: int, config: Config) -> Agent:

    agent_dict = {
        'nn_mil': MILNNAgent,
        'hdc_mil': MILHDCAgent,
        'nn': MLPNNAgent
    }

    try:
        return agent_dict[config.type_agent](state_space_size, action_space_size, config)
    except KeyError:
        raise TypeError(f'The model type "{config.type_agent}" is not supported.')
