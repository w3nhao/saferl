import os
import logging
import torch
import json
from typing import Optional
from typing import Tuple
from datetime import datetime

from data.dataset import SequenceDataset
from model.unet import Unet1D
from model.diffusion import GaussianDiffusion
from model.trainer import Trainer
from utils.common import set_seed, save_config, setup_logging, build_model
from dsrl.infos import DENSITY_CFG
from configs.pretrain_config import TrainConfig, get_train_config, TASK_CONFIG

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym  # noqa

from IPython import embed

# new version
def setup_experiment(config: TrainConfig) -> Tuple[str, str]:
    """Setup experiment directory and save metadata
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple[str, str]: Paths to experiment and model directories
    """
    # Create directories
    exp_dir = os.path.join(config.experiments_dir, f'{config.exp_id}_{config.dim}')
    model_dir = os.path.join(config.checkpoints_dir, f'{config.exp_id}_{config.dim}')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    serializable_config = {}
    for key, value in config.__dict__.items():
        if value is None:
            serializable_config[key] = "null" 
        elif isinstance(value, torch.Tensor):
            serializable_config[key] = value.tolist() if value.numel() > 0 else [] 
        else:
            try:
                json.dumps(value)
                serializable_config[key] = value
            except (TypeError, OverflowError):
                serializable_config[key] = str(value)
    
    # Save experiment config
    save_config(serializable_config, os.path.join(exp_dir, 'config.json'))
    
    # Update pretrain metadata
    metadata_path = os.path.join(config.experiments_dir, 'metadata/pretrain.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    metadata = {}
    # if os.path.exists(metadata_path):
    #     with open(metadata_path, 'r') as f:
    #         metadata = json.load(f)
    metadata[config.exp_id] = {
        'date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'description': config.description if hasattr(config, 'description') else '',
        'config': serializable_config
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return exp_dir, model_dir

def train(config: TrainConfig):
    """Main training function
    
    Args:
        config: Training configuration
        device: Device to place model and data on
    """
    # Setup
    config = TASK_CONFIG[config.task]()
    exp_dir, model_dir = setup_experiment(config)
    setup_logging(exp_dir)
    set_seed(config.seed)
    config.scaler = torch.tensor(config.scaler).reshape(-1, 1)

    # setup device
    torch.cuda.set_device(config.gpu_id)
    config.device = torch.device(f"cuda:{config.gpu_id}")
    logging.info(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    
    # Load dataset
    # initialize environment
    if "Metadrive" in config.task:
        import gym
    import gymnasium as gym  # noqa
    env = gym.make(config.task)

    data = env.get_dataset()
    env.set_target_cost(config.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if config.density != 1.0:
        density_cfg = DENSITY_CFG[config.task + "_density" + str(config.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(data,
                                config.outliers_percent,
                                config.noise_scale,
                                config.inpaint_ranges,
                                config.epsilon,
                                config.density,
                                cbins=cbins,
                                rbins=rbins,
                                max_npb=max_npb,
                                min_npb=min_npb)
    dataset = SequenceDataset(
        data,
        scaler=config.scaler,
        split='train'
    )
    logging.info(f'Sample shape: {dataset.shape}')
    # Build model
    model = build_model(config, dataset)
    
    # Setup trainer
    trainer = Trainer(
        model,
        dataset,
        results_folder=model_dir,
        train_num_steps=config.train_num_steps,
        save_and_sample_every=config.checkpoint_interval,
        train_lr=config.lr,
    )
    
    # Train
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed")

def main():
    """Main entry point"""
    
    config = get_train_config(exp_id="turbo_MultiScaler", model_size="turbo")
    
    train(config)

if __name__ == "__main__":
    main()
