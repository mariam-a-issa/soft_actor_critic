from dataclasses import dataclass, field, replace, asdict
from typing import Optional, Dict
import json

@dataclass
class Config:
    hdc_agent: bool = True
    mil_agent: bool = True
    alpha_value: float = 0.4
    autotune: bool = True
    alpha_lr: float = 3e-4
    critic_lr: float = 3e-4
    policy_lr: float = 3e-4
    hypervec_dim: int = 4096
    hidden_size: int = 128
    pos_enc_size: int = 8
    grad_clip: int = 5
    sample_size: int = 64
    tau: float = 0.005
    target_update: int = 1
    seed: Optional[int] = None
    explore_steps: int = 0
    buffer_size: int = 10**6
    learning_steps: int = 1
    update_frequency: int = 1
    environment_info: Dict = field(default_factory=lambda: {
        'id': 'NASimEmu-v0',
        'emulate': False,
        'scenario_name': '/home/ian/projects/hd_sac/NetworkAttackSimulator/nasim/scenarios/benchmark/medium.yaml',
        'step_limit': 100,
        'augment_with_action': True
    })
    max_steps: int = 250000
    eval_frequency: int = 10
    num_evals: int = 5
    tensorboard: bool = False
    wandb: bool = False
    dynamic: bool = True
    target_start: float = 0.8
    target_end: float = 0.2
    midpoint: float = 0.45
    slope: float = 6
    attention: bool = False
    num_heads: int = 2
    graph: bool = False
    messages_passed: int = 2
    gpu: bool = True
    discount : float = .99
    save_csv : bool = False

    def with_updates(self, **entries):
        return replace(self, **entries)
    
    def to_flat_dict(self):
        config_dict = asdict(self)
        for key in list(config_dict.keys()):  # Iterate over keys explicitly
            if isinstance(config_dict[key], dict):
                config_dict.update(config_dict.pop(key))  # Unpack and remove the original key
        return config_dict
    
    def update_from_json(self, json_path):
        with open(json_path, 'r') as file:
            updates = json.load(file)
        return self.with_updates(**updates)