import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

@dataclass
class TrainConfig:
    """Training configuration for diffusion models."""
    
    # Basic settings
    exp_id: str                           # Experiment folder id
    seed: int = 42                        # Random seed for reproducibility
    date_time: str = datetime.today().strftime('%Y-%m-%d')  # Date for experiment folder
    device: str = "cuda"                  # Device to use
    gpu_id: int = 1                      # GPU ID to use

    # Training settings
    train_num_steps: int = 100000         # Total number of training steps
    checkpoint_interval: int = 1000      # Save checkpoint every N steps
    lr: float = 1e-5                      # Learning rate for training
    scaler: List[float] = (1,)
    
    # Dataset settings
    task: str = "OfflinePointPush1Gymnasium-v0"    # Dataset name for training
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    density: float = 1.0

    # evaluation params
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((450.0, 10), (500.0, 20), (550.0, 50))  # reward, cost
    cost_limit: int = 10

    # UNet hyperparameters
    dim: int = 256                         # Base dimension for UNet features
    resnet_block_groups: int = 1          # Number of groups in GroupNorm
    dim_mults: List[int] = (1, 2, 4, 8)   # Channel multipliers for each UNet level
    
    # Conditioning settings
    is_condition_u0: bool = True         # Whether to learn p(u_[1, T] | u0)
    is_condition_u0_zero_pred_noise: bool = True  # Enforce zero pred_noise for conditioned data when learning p(u_[1, T-1] | u0)
    
    # Sampling settings
    using_ddim: bool = False             # Whether to use DDIM sampler
    ddim_eta: float = 0.0                # DDIM eta parameter
    ddim_sampling_steps: int = 1000      # Number of DDIM sampling steps
    
    @property
    def base_dir(self) -> str:
        """Get the base directory path."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @property
    def datasets_dir(self) -> str:
        """Get the absolute datasets directory path."""
        return '/tokamak_data/consolidated_dataset'
    
    @property
    def experiments_dir(self) -> str:
        """Get the experiments directory path."""
        return os.path.join(self.base_dir, "experiments")
    
    @property
    def checkpoints_dir(self) -> str:
        """Get the model checkpoints directory path."""
        return os.path.join(self.experiments_dir, "checkpoints")

def get_train_config(exp_id: str, model_size: str) -> TrainConfig:
    """Get configuration template based on some input values."""
    if model_size == "turbo":
        return TrainConfig(
            exp_id=exp_id,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            train_num_steps=200000,
        )
    else:
        return TrainConfig(
            exp_id=exp_id,
        ) 

@dataclass
class CarGoal1Config(TrainConfig):
    # model params
    episode_len: int = 1000
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    state_channel: int = 72
    scaler: List[float] = (18, 21, 53,  0.5,  1.5,  0.5,  4,  4,
         4,  0.5000,  0.5000,  0.2, 16, 14, 15,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1.2000,  1)
    target_returns: Tuple[Tuple[float, ...], ...] = ((40.0, 20), (40.0, 40), (40.0, 80))
    gpu_id: int = 2
    exp_id: str = "car_goal1"
    state_channel: int = 72

@dataclass
class PointGoal1Config(TrainConfig):
    # model params
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    state_channel: int = 60
    scaler: List[float] = (6, 20,  10,  2,  1.5,  1.0000,  1.0000,  1.0000,
         4,  0.5000,  0.5000,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1.2,  1)
    target_returns: Tuple[Tuple[float, ...], ...] = ((30.0, 20), (30.0, 40), (30.0, 80))
    gpu_id: int = 4
    exp_id: str = "point_goal1"


@dataclass
class PointPush1Config(TrainConfig):
    # model params
    episode_len: int = 1000
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    state_channel: int = 76
    scaler: List[float] = (135,  82,   10,   1.5,   1,   1,   1,
          1,   4,   0.5000,   0.5000,   1,   1,   1,
          1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,
          1,  1,  1,  1,  1,  1,  1,
         1,   1.2,  1)
    target_returns: Tuple[Tuple[float, ...], ...] = ((15.0, 20), (15.0, 40), (15.0, 80))
    gpu_id: int = 5
    exp_id: str = "point_push1"

TASK_CONFIG = {
    "OfflineCarGoal1Gymnasium-v0": CarGoal1Config,
    "OfflinePointGoal1Gymnasium-v0": PointGoal1Config,
    "OfflinePointPush1Gymnasium-v0": PointPush1Config,
}
