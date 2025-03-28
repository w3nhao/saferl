import torch
import argparse
import json
import logging
from datetime import datetime
import os

from data.dataset import SequenceDataset
from configs.inference_config import InferenceConfig, TASK_CONFIG
from inference.pipeline import InferencePipeline
from dsrl.infos import DENSITY_CFG
from utils.common import set_seed, setup_logging, load_model
from dsrl.infos import DENSITY_CFG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--exp_id', type=str, default="turbo", help='Experiment ID')
    parser.add_argument('--tuning_dir', type=str, default="finetune", help='Tuning directory')
    parser.add_argument('--tuning_id', type=str, default="test", help='Tuning ID')
    parser.add_argument('--task', type=str, default='OfflinePointPush1Gymnasium-v0', help='Task')
    parser.add_argument('--dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--checkpoint', type=int, required=True, help='Checkpoint to use')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help='Checkpoint directory')
    parser.add_argument('--post_train_id', type=str, default="total", help='Post train ID')
    parser.add_argument('--finetune_set', type=str, default="train", help='Mode of finetune set: train, test')
    parser.add_argument('--wo_post_train', action='store_true', help='No post train')
    parser.add_argument('--use_guidance', action='store_true', help='use guidance')
    parser.add_argument('--backward_finetune', action='store_true', help='backward finetune')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha of conformal prediction')
    parser.add_argument('--finetune_epoch', type=int, required=True, help='Number of finetune epochs')
    parser.add_argument('--finetune_steps', type=int, required=True, help='Number of finetune steps')
    parser.add_argument('--finetune_lr', type=float, required=True, help='Finetune learning rate')
    parser.add_argument('--guidance_weights', type=str, default=None, help='JSON string of guidance weights')
    parser.add_argument('--loss_weights', type=str, default=None, help='JSON string of loss weights')
    parser.add_argument('--guidance_scaler', type=float, default=1.0, help='Guidance scaler')
    parser.add_argument('--train_batch_size', type=int, default=1000, help='Training batch size')
    parser.add_argument('--cal_batch_size', type=int, default=1000, help='Calibration batch size')
    parser.add_argument('--ddim_sampling_steps', type=int, default=200, help='DDIM sampling steps')
    parser.add_argument('--ddim_eta', type=float, default=1.0, help='DDIM sampling eta')
    args = parser.parse_args()

    # basic config
    config = TASK_CONFIG[args.task]()

    if args.finetune_set == 'test':
        config.finetune_steps = 1
    if args.finetune_set == 'test' and args.use_guidance == False:
        args.guidance_scaler = 0
    if args.loss_weights:
        config.loss_weights = json.loads(args.loss_weights)
    if args.guidance_weights:
        config.guidance_weights = json.loads(args.guidance_weights)

    for attr in ['checkpoint', 'checkpoint_dir', 'wo_post_train', 'post_train_id', 
                'exp_id', 'tuning_dir', 'tuning_id', 'gpu_id', 'dim', 'finetune_set',
                'alpha', 'finetune_epoch', 'finetune_steps', 'finetune_lr', 
                'train_batch_size', 'cal_batch_size', 'ddim_sampling_steps', 'ddim_eta',
                'use_guidance', 'guidance_scaler', 'backward_finetune']:
        setattr(config, attr, getattr(args, attr))

    config.scaler = torch.tensor(config.scaler).reshape(-1, 1)

    log_dir = os.path.join(
        config.experiments_dir,
        config.exp_id,
        config.tuning_dir,
        config.tuning_id
    )
    setup_logging(log_dir)
    
    torch.cuda.set_device(config.gpu_id)
    config.device = torch.device(f"cuda:{config.gpu_id}")
    logging.info(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    
    set_seed(config.seed)
    
    if "Metadrive" in config.task:
        import gym
    else:
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
    cal_dataset = SequenceDataset(
        data,
        scaler=config.scaler,
        split='cal'
    )   # only used for loading trainer
    model, _ = load_model(config, cal_dataset)

    pipeline = InferencePipeline(config, model)
    results = pipeline.run()

    logging.info("Done!")
    
if __name__ == '__main__':
    main()