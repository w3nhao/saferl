import os
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from pathlib import Path

@dataclass
class InferenceConfig:
    """Configuration for inference
    
    Args:
  
    """
    # Experiment settings
    tuning_dir: str = "finetune"
    tuning_id: str = "1"
    exp_id: str = "turbo_MultiScaler"
    seed: int = 42
    gpu_id: int = 0
    device: str = "cuda"
    
    # Dataset settings
    task: str = "OfflinePointPush1Gymnasium-v0"    # Dataset name for training
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    nt_total: int = 1000
    pad_size: int = 1024
    safety_threshold: float = 0
    reward_scale: float = 0.1
    cost_scale: float = 1
    cost_reverse: bool = False

    # evaluation params
    target_returns: Tuple[Tuple[float, ...],
                          ...] = ((450.0, 10), (500.0, 20), (550.0, 50))  # reward, cost
    cost_limit: int = 10
    eval_episodes: int = 20
    test_batch_size: int = 20

    n_cal_samples: int = 100
    cal_batch_size: int = 100
    num_cal_batch: int = 1

    train_batch_size: int = 1000    # batch_size of training data in per finetune step
    
    # finetuning settings
    finetune_set: str = "train"
    use_guidance: bool = False
    backward_finetune: bool = False
    optimizer: str = "adam"
    finetune_lr: float = 1e-5
    finetune_epoch: int = 2    # number of finetune epochs
    finetune_steps: int = 2   # number of finetune steps in one epoch
    loss_weights: Optional[dict] = None
    use_grad_norm: bool = False

    # conformal settings
    alpha: float = 0.9

    # Load model settings
    checkpoint_dir: str = "checkpoints"
    wo_post_train: bool = True
    post_train_id: str = "total"
    checkpoint: int = 190
    train_num_steps: int = 200000
    checkpoint_interval: int = 1000

    # Diffusion settings
    using_ddim: bool = True
    ddim_eta: float = 1.0
    ddim_sampling_steps: int = 200
    J_scheduler: str = "constant"
    w_scheduler: str = "constant"
             
    # Guidance settings
    guidance_weights: Optional[dict] = None
    guidance_scaler: Optional[float] = None
    
    # Model conditioning
    is_condition_u0: bool = True
    is_condition_u0_zero_pred_noise: bool = True
    
    # UNet settings
    dim: int = 256
    resnet_block_groups: int = 1
    dim_mults: Optional[List[int]] = None

    @property
    def base_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @property
    def datasets_dir(self) -> str:
        return os.path.join(self.base_dir, "datasets")
    
    @property
    def experiments_dir(self) -> str:
        return os.path.join(self.base_dir, "experiments")
    
    @property
    def checkpoints_dir(self) -> str:
        if self.finetune_set == "train" or self.wo_post_train:
            return os.path.join(self.experiments_dir, self.checkpoint_dir)
        else:
            return os.path.join(self.experiments_dir, self.exp_id, self.post_train_id)

    def __post_init__(self):
        """Set default values that depend on other fields"""        
        if self.loss_weights is None:
            self.loss_weights = {
                "loss_train": 1.0,
            }

        if self.guidance_weights is None:
            self.guidance_weights = {
                "w_obj": 1.0,
                "w_safe": 1.0,
            }
        if self.guidance_scaler is None:
            self.guidance_scaler = 1.0
            
        if self.dim_mults is None:
            self.dim_mults = [1, 2, 4, 8]

# def get_inference_config(model_size: str = "large", exp_id: str = "turbo", tuning_dir: str = "finetune", tuning_id: str = "test") -> InferenceConfig:
#     """Get evaluation configuration"""
#     if model_size == "large":
#         return InferenceConfig(
#             exp_id=exp_id,
#             tuning_dir=tuning_dir,
#             tuning_id=tuning_id,
#             dim=512,
#             dim_mults=(1, 2, 4, 8),
#         )
#     elif model_size == "turbo":
#         return InferenceConfig(
#             exp_id=exp_id,
#             tuning_dir=tuning_dir,
#             tuning_id=tuning_id,
#             dim=256,
#             dim_mults=(1, 2, 4, 8),
#         )



@dataclass
class CarGoal1Config(InferenceConfig):
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
class PointGoal1Config(InferenceConfig):
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
class PointPush1Config(InferenceConfig):
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
