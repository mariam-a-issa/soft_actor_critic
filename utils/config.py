from dataclasses import dataclass, field, replace, asdict
from typing import Optional, Dict, Literal
import json



@dataclass
class Config:
    """
    Configuration class for training, model, learning, environment, logging, and architecture parameters.
    """

    ### RL Parameters ###
    alpha_value: float = 0.4
    """Initial value of the temperature parameter for entropy regularization"""
    
    autotune: bool = True
    """Whether to automatically adjust the temperature parameter (alpha)"""
    
    discount: float = 0.99
    """Discount factor for future rewards (gamma)"""
    
    tau: float = 0.005
    """Soft update parameter for target networks"""
    
    target_update: int = 1
    """Frequency (in steps) of target network updates"""

    ### Target Entropy Parameters (for Alpha Update) ###
    target_entropy_end: float = 0.2
    """Final target entropy value (after annealing)"""
    
    target_entropy_midpoint: float = 0.4
    """Midpoint of the sigmoid function controlling entropy decay"""
    
    target_entropy_slope: float = 7
    """Slope of the entropy decay sigmoid function"""
    
    target_entropy_start: float = 0.8
    """Initial target entropy value"""

    ### Model Parameters ###
    alpha_lr: float = 3e-4
    """Learning rate for the temperature parameter (alpha)"""
    
    critic_lr: float = 3e-4
    """Learning rate for the critic network"""
    
    hidden_dim: int = 64
    """Number of units in hidden layers of neural networks"""
    
    hypervec_dim: int = 4096
    """Dimensionality of hypervector representations (if applicable)"""
    
    policy_lr: float = 3e-4
    """Learning rate for the policy network"""
    
    pos_enc_dim: int = 8
    """Size of positional encoding (if used)"""
    
    type_agent: Literal['hdc', 'nn', 'nn_mil', 'hdc_mil'] = 'hdc_agent'
    """Type of agent to use: 
    - 'hdc' (Hyperdimensional Computing-based with padding)
    - 'nn' (Standard neural network with padding)
    - 'nn_mil' (Neural network with MIL)
    - 'hdc_mil' (HDC-based model with MIL)
    """

    ### Learning Parameters ###
    buffer_size: int = 10**6
    """Size of the experience replay buffer"""
    
    explore_steps: int = 0
    """Number of initial random exploration steps before learning begins"""
    
    grad_clip: Optional[int] = 5
    """Maximum gradient norm for clipping critic network"""
    
    learning_steps: int = 1
    """Number of gradient updates per training step"""
    
    max_steps: int = 200000
    """Total number of environment steps during training"""
    
    sample_size: int = 64
    """Number of samples per batch for training"""
    
    seed: Optional[int] = None
    """Random seed for reproducibility"""
    
    update_frequency: int = 1
    """Frequency of policy updates relative to environment steps"""

    ### Environment Parameters ###
    environment_info: Dict = field(default_factory=lambda: {
        'id': 'NASimEmu-v0',
        'augment_with_action': True,
        'emulate': False,
        'scenario_name': '/home/ian/projects/hd_sac/NetworkAttackSimulator/nasim/scenarios/benchmark/medium.yaml',
        'step_limit': 100
    })
    """Dictionary containing environment parameters"""
    
    ### Logging Parameters ###
    eval_frequency: int = 10
    """Number of training steps between evaluations"""
    
    num_evals: int = 5
    """Number of evaluation episodes per evaluation cycle"""
    
    save_csv: bool = False
    """Whether to log results to a CSV file"""
    
    tensorboard: bool = False
    """Whether to log metrics to TensorBoard"""
    
    wandb: bool = False
    """Whether to use Weights & Biases for logging"""
    
    wandb_project_name : str = "SAC in MIL"
    """The name of the current project"""
    
    num_sha_char : int = 7
    """The number of characters from the sha to show on the name of a wandb experiment"""

    ### Architecture Parameters ###
    attention: bool = False
    """Whether to use an attention mechanism in the model"""
    
    gpu: bool = True
    """Whether to use GPU for training"""

    gpu_device: int = 0
    """What GPU to train on"""
    
    graph: bool = False
    """Whether to use a graph-based model"""
    
    messages_passes: int = 2
    """Number of message-passing iterations (if using a graph-based model)"""
    
    num_heads: int = 2
    """Number of attention heads (if using attention)"""

    def with_updates(self, **entries):
        """Return a new Config object with updated values."""
        return replace(self, **entries)
    
    def to_flat_dict(self):
        """Convert the configuration to a flat dictionary, merging nested dictionaries."""
        config_dict = asdict(self)
        for key in list(config_dict.keys()):  # Iterate over keys explicitly
            if isinstance(config_dict[key], dict):
                config_dict.update(config_dict.pop(key))  # Unpack and remove the original key
        return config_dict
    
    def update_from_json(self, json_path):
        """Load configuration updates from a JSON file and return a new Config object with the updates applied."""
        with open(json_path, 'r') as file:
            updates = json.load(file)
        return self.with_updates(**updates)