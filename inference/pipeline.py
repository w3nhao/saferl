import os
import json
import torch
from typing import Dict, Optional, Tuple
import logging
import time
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from itertools import cycle

from data.dataset import SequenceDataset
from configs.inference_config import InferenceConfig
from .conformal import ConformalCalculator
from .utils import get_scheduler
from utils.common import get_target
from utils.guidance import calculate_weight, get_gradient_guidance, normalize_weights
from utils.metrics import evaluate_samples, control_trajectories, calculate_safety_score


class InferencePipeline:
    """Main pipeline class for inference and fine-tuning
    
    Key functionalities:
    1. Model inference with guidance
    2. Conformal prediction
    3. Model fine-tuning with safety constraints
    4. Metrics evaluation
    """
    def __init__(self, config: InferenceConfig, model):
        self.config = config
        self.device = self.config.device
        self.model = model.to(self.config.device)
        
        # setup data
        self.setup_data()
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
        self.data = env.pre_process_data(data,
                                    config.outliers_percent,
                                    config.noise_scale,
                                    config.inpaint_ranges,
                                    config.epsilon,
                                    config.density,
                                    cbins=cbins,
                                    rbins=rbins,
                                    max_npb=max_npb,
                                    min_npb=min_npb)
        self.max_action = env.action_space.high[0]
        
        # Initialize optimizer
        self.setup_optimizer()
        
        # Initialize conformal calculator
        self.conformal_calculator = ConformalCalculator(self.model, self.config)
        
        # Setup guidance
        self.setup_guidance()
        self.J_scheduler = get_scheduler(self.config.J_scheduler)
        self.w_scheduler = get_scheduler(self.config.w_scheduler)
        
        # Initialize quantile value
        self.Q = 0  # Initialize quantile
        
    def evaluate(self):
        """
        Evaluates the performance of the model on a number of episodes.
        """
        self.model.eval()
        episode_rets, episode_costs, episode_lens = [], [], []
        for _ in trange(self.config.eval_episodes, desc="Evaluating...", leave=False):
            epi_ret, epi_len, epi_cost = self.rollout(self.model, self.env)
            episode_rets.append(epi_ret)
            episode_lens.append(epi_len)
            episode_costs.append(epi_cost)
        self.model.train()
        return np.mean(episode_rets) / self.reward_scale, np.mean(
            episode_costs) / self.cost_scale, np.mean(episode_lens)

    @torch.no_grad()
    def rollout(
        self,
        model: CDT,
        env: gym.Env,
        target_return: float,
        target_cost: float,
    ) -> Tuple[float, float]:
        """
        Evaluates the performance of the model on a single episode.
        """
        states = torch.zeros(1,
                             model.episode_len + 1,
                             model.state_dim,
                             dtype=torch.float,
                             device=self.device)
        actions = torch.zeros(1,
                              model.episode_len,
                              model.action_dim,
                              dtype=torch.float,
                              device=self.device)
        returns = torch.zeros(1,
                              model.episode_len + 1,
                              dtype=torch.float,
                              device=self.device)
        costs = torch.zeros(1,
                            model.episode_len + 1,
                            dtype=torch.float,
                            device=self.device)
        time_steps = torch.arange(model.episode_len,
                                  dtype=torch.long,
                                  device=self.device)
        time_steps = time_steps.view(1, -1)

        obs, info = env.reset()
        states[:, 0] = torch.as_tensor(obs, device=self.device)
        returns[:, 0] = torch.as_tensor(target_return, device=self.device)
        costs[:, 0] = torch.as_tensor(target_cost, device=self.device)

        epi_cost = torch.tensor(np.array([target_cost]),
                                dtype=torch.float,
                                device=self.device)

        # cannot step higher than model episode len, as timestep embeddings will crash
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for step in range(self.config.episode_len):
            # first select history up to step, then select last seq_len states,
            # step + 1 as : operator is not inclusive, last action is dummy with zeros
            # (as model will predict last, actual last values are not important) # fix this noqa!!!
            s = states[:, :step + 1][:, -model.seq_len:]  # noqa
            a = actions[:, :step + 1][:, -model.seq_len:]  # noqa
            r = returns[:, :step + 1][:, -model.seq_len:]  # noqa
            c = costs[:, :step + 1][:, -model.seq_len:]  # noqa
            t = time_steps[:, :step + 1][:, -model.seq_len:]  # noqa

            acts, _, _ = model(s, a, r, c, t, None, epi_cost)
            if self.stochastic:
                acts = acts.mean
            acts = acts.clamp(-self.max_action, self.max_action)
            act = acts[0, -1].cpu().numpy()
            # act = self.get_ensemble_action(1, model, s, a, r, c, t, epi_cost)

            obs_next, reward, terminated, truncated, info = env.step(act)
            if self.cost_reverse:
                cost = (1.0 - info["cost"]) * self.cost_scale
            else:
                cost = info["cost"] * self.cost_scale
            # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
            actions[:, step] = torch.as_tensor(act)
            states[:, step + 1] = torch.as_tensor(obs_next)
            returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)
            costs[:, step + 1] = torch.as_tensor(costs[:, step] - cost)

            obs = obs_next

            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]

            if terminated or truncated:
                break

        return episode_ret, episode_len, episode_cost

    def setup_data(self):
        """Setup datasets and dataloaders"""
        logging.info("Setting up datasets and loaders...")
        
        # Train dataset
        self.train_dataset = SequenceDataset(
            self.data,
            scaler=self.config.scaler,
            split="train",
            is_normalize=True,
            is_need_idx=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )
        self.train_loader_iter = cycle(self.train_loader)
        
        # # Pre-cache all target states
        # logging.info("Pre-caching target states...")
        # start = time.time()
        # loader = DataLoader(
        #     self.train_dataset,
        #     batch_size=4096,
        #     num_workers=32,
        #     pin_memory=True,
        #     shuffle=False
        # )
        # targets = []
        # for batch, idx in loader:
        #     targets.append(batch)
        # self.train_targets = torch.cat(targets, dim=0)[:, :3, :self.train_dataset.nt_total]
        # logging.info(f"Train targets cached, shape: {self.train_targets.shape}, time: {time.time() - start:.2f}s")
        
        # Calibration dataset
        self.cal_dataset = SequenceDataset(
            self.data,
            reward_scale=self.config.reward_scale,
            cost_scale=self.config.cost_scale,
            split="cal",
            is_normalize=True,
            is_need_idx=True,
        )
        self.cal_loader = DataLoader(
            self.cal_dataset,
            batch_size=self.config.cal_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        self.cal_loader_iter = cycle(self.cal_loader)
        
        # # Cache cal targets
        # start = time.time()
        # loader = DataLoader(
        #     self.cal_dataset,
        #     batch_size=4096,
        #     num_workers=32,
        #     pin_memory=True,
        #     shuffle=False
        # )
        # targets = []
        # for batch, idx in loader:
        #     targets.append(batch)
        # self.cal_targets = torch.cat(targets, dim=0)[:, :3, :self.cal_dataset.nt_total]
        # logging.info(f"Cal targets cached, shape: {self.cal_targets.shape}, time: {time.time() - start:.2f}s")
        
        # # Test dataset
        # self.test_dataset = SequenceDataset(
        #     self.data,
        #     scaler=self.config.scaler,
        #     split="test",
        #     is_normalize=True,
        #     is_need_idx=True,
        # )
        # self.test_loader = DataLoader(
        #     self.test_dataset,
        #     batch_size=self.config.test_batch_size,
        #     shuffle=False,
        #     num_workers=4,
        #     pin_memory=True
        # )
        
        # # Cache test targets
        # start = time.time()
        # loader = DataLoader(
        #     self.test_dataset,
        #     batch_size=4096,
        #     num_workers=32,
        #     pin_memory=True,
        #     shuffle=False
        # )
        # targets = []
        # for batch, idx in loader:
        #     targets.append(batch)
        # self.test_targets = torch.cat(targets, dim=0)[:, :3, :self.test_dataset.nt_total]
        # logging.info(f"Test targets cached, shape: {self.test_targets.shape}, time: {time.time() - start:.2f}s")
        
        logging.info(f"Train batch size: {self.config.train_batch_size}")
        logging.info(f"Calibration batch size: {self.config.cal_batch_size}")
        logging.info(f"Test batch size: {self.config.test_batch_size}")
        
    def setup_optimizer(self):
        """Configure optimizer"""
        if self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.finetune_lr,
                betas=(0.99, 0.999)
            )
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.finetune_lr,
                momentum=0.9
            )
        
    def setup_guidance(self):
        """Setup guidance function for test sampling"""
        self.guidance_fn = lambda x: get_gradient_guidance(x, 
            data=self.data,
            scaler=self.config.scaler,
            target_i=list(range(self.config.n_test_samples)),
            w_obj=self.config.guidance_weights["w_obj"],
            w_safe=self.config.guidance_weights["w_safe"],
            guidance_scaler=self.config.guidance_scaler,
            nt=self.config.nt_total,
            safety_threshold=self.config.safety_threshold,
            device=self.device,
        ) if any(self.config.guidance_weights.values()) else None
        
        self.J_scheduler = get_scheduler(self.config.J_scheduler)
        self.w_scheduler = get_scheduler(self.config.w_scheduler)

    def get_finetune_reweights(self, data_loader, mode):
        """Get reweights for fine-tuning loss"""
        weights = []
        batch_size = data_loader.batch_size
        n_samples = len(data_loader.dataset)
        
        if mode == "train":
            state_target = self.train_targets.to(self.device)
        else:
            state_target = self.test_targets.to(self.device)
        
        for i, (state, idx) in tqdm(enumerate(data_loader), desc="Getting finetune reweights"):
            state = state.to(self.device)
            idx = idx.to(self.device)
            state_target_batch = state_target[idx]
            weight = calculate_weight(state, 
                                    state_target_batch, 
                                    self.config.scaler,
                                    self.config.nt_total, 
                                    self.Q, 
                                    self.config.safety_threshold, 
                                    self.config.guidance_weights['w_obj'], 
                                    self.config.guidance_weights['w_safe'], 
                                    self.config.guidance_scaler)
            weights.append(weight)
        weights = torch.cat(weights)
        normalized_weights = normalize_weights(weights)

        return normalized_weights

    def finetune_step(self, train_state, prediction, reweights_train, reweights_test):
        """design loss function and finetune model with weighted loss
        """
        self.model.train()
        train_state, prediction = train_state.to(self.device), prediction.to(self.device)
        reweight_train, reweight_test = reweights_train.to(self.device), reweights_test.to(self.device)

        # Calculate loss based on training set
        loss_diff_train = self.model(train_state, mean=False)
        loss_train = (reweight_train * loss_diff_train).mean()
        loss_train.backward()

        # Calculate loss based on test sampling
        if self.config.loss_weights['loss_test'] > 0:
            loss_diff_test = self.model(prediction, mean=False)
            loss_test = (reweight_test * loss_diff_test).mean()
        else: 
            loss_diff_test = torch.zeros_like(reweight_test)
            loss_test = torch.tensor(0.)

        loss = self.config.loss_weights['loss_train'] * loss_train + self.config.loss_weights['loss_test'] * loss_test

        # logging.info(f"Diff loss: {loss_diff_train.mean().item():.4f}, {loss_diff_test.mean().item():.4f}")

        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {'loss': loss.item(), 'loss_train': loss_train.item(), 'loss_test': loss_test.item()}

    def backward_finetune_step(self, prediction, state_target):
        """design loss function and finetune model with backward loss
        """
        self.model.train()

        state = prediction[:, :3, :self.config.nt_total]

        beta_p_final = state[:, 0, :]
        l_i_final = state[:, 2, :]

        beta_p_final_gt = state_target[:, 0, :]
        l_i_final_gt = state_target[:, 2, :]
        
        objective_beta_p = (beta_p_final - beta_p_final_gt).square().mean(-1)
        objective_l_i = (l_i_final - l_i_final_gt).square().mean(-1)
        objective = objective_beta_p + objective_l_i

        s = calculate_safety_score(state)
        safe_cost = torch.maximum(
            self.config.safety_threshold - s + self.Q,
            torch.zeros_like(s)
        )

        loss = (self.config.guidance_weights['w_obj'] * objective + \
            self.config.guidance_weights['w_safe'] * safe_cost).mean()
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {'loss': loss.item()}

    def run_epoch(self, epoch) -> Dict:
        """Run one epoch of fine-tuning and evaluation
        
        Returns:
            Dict: Metrics for this epoch
        """
        # if epoch != 0:
        quantile = self.calibrate()
        self.Q = quantile
        # self.Q = 0

        logging.info(f"Calculating reweights for training and test")

        if not self.config.backward_finetune:
            reweights_training = self.get_finetune_reweights(self.train_loader, mode="train")
            reweights_test = self.get_finetune_reweights(self.test_loader, mode="test")
            logging.info(f"Reweights calculated")

        all_prediction = []
        train_metrics = []
        for test_state, idx in self.test_loader:
            state_target_batch = self.test_targets[idx].to(self.device)
            if self.config.loss_weights['loss_test'] > 0 or self.config.backward_finetune:
                prediction = self.inference(test_state, backward=self.config.backward_finetune)
            else:
                prediction = torch.zeros_like(test_state)
            all_prediction.append(prediction)

            logging.info(f"Finetuning {self.config.finetune_steps} steps")
            for finetune_step in range(self.config.finetune_steps):
                if not self.config.backward_finetune:
                    train_batch, sim_id = next(self.train_loader_iter)
                    batch_loss = self.finetune_step(train_batch, prediction, reweights_training[sim_id], reweights_test)
                else:
                    batch_loss = self.backward_finetune_step(prediction, state_target_batch)
                train_metrics.append(batch_loss)
                if finetune_step % 100 == 0:
                    logging.info(f"Finetuning step {finetune_step}, loss: {batch_loss['loss']:.4f}")
            if not self.config.backward_finetune:
                break

        # Calculate average training metrics
        avg_train_metrics = {}
        if len(train_metrics) > 0:  
            for key in train_metrics[0].keys():
                avg_train_metrics[key] = sum(m[key] for m in train_metrics) / len(train_metrics)
        
        eval_metrics = self.evaluate_model()
        
        # Merge all metrics
        epoch_metrics = {
            'train': avg_train_metrics,
            'eval': eval_metrics,
            'quantile': self.Q.item() if isinstance(self.Q, torch.Tensor) else self.Q
        }
        
        return epoch_metrics

    def evaluate_model(self) -> Dict:
        """Evaluate current model on test dataset
        
        Returns:
            Dict: Evaluation metrics
        """
        logging.info("Starting evaluation...")
        all_predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for test_state, idx in self.test_loader:
                test_state = test_state.to(self.device)
                
                # Generate samples
                # CHOICE: None or self.guidance_fn. reweighted loss can replace guidance
                predictions = self.inference(test_state)
                
                all_predictions.append(predictions)
        
        # Concatenate all results
        predictions = torch.cat(all_predictions)
        state_controlled = control_trajectories(predictions, self.config.nt_total, self.config.seed)
        state_target = get_target(list(range(self.config.n_test_samples)), data=self.data, scaler=self.config.scaler, 
                                is_normalize=False).to(self.config.device)
        
        metrics = evaluate_samples(
            predictions, 
            state_controlled, 
            state_target, 
            self.config.safety_threshold,
            self.test_dataset
        )
        
        logging.info("Evaluation completed")
        return metrics

    def calibrate(self) -> float:
        """Run calibration phase
        
        Returns:
            float: Calibrated quantile value
        """
        logging.info("Starting calibration phase...")
        
        # Get conformal scores
        scores, weights, states = self.conformal_calculator.get_conformal_scores(self.cal_loader_iter, self.cal_targets, self.Q)
        
        # Calculate quantile
        quantile = self.conformal_calculator.calculate_quantile(
            scores, weights, states, self.config.alpha
        )
        
        logging.info(f"Calibration completed. Quantile: {quantile:.4f}")
        return quantile
        
    def inference(self, state: torch.Tensor, backward: bool = False) -> Tuple[torch.Tensor, Dict]:
        """Run inference phase
               
        Returns:
            Tuple[torch.Tensor, Dict]: Predictions and metrics
        """ 
        state = state.to(self.device)
        # Get w
        if self.config.use_guidance:
            nablaJ=self.guidance_fn
        else:
            nablaJ=None
        output = self.model.sample(
            batch_size=state.shape[0],
            clip_denoised=True,
            guidance_u0=True,
            device=self.device,
            u_init=state[:, :self.config.state_channel, 0],
            nablaJ=nablaJ,  
            J_scheduler=self.J_scheduler,
            w_scheduler=self.w_scheduler,
            enable_grad=backward,
        )
        pred = output * self.config.scaler.to(self.device)

        return pred
    
    def run(self) -> Dict:
        """Run multiple training epochs and record results"""
        # Setup finetune directory
        finetune_dir = os.path.join(
            self.config.experiments_dir,
            self.config.exp_id,
            self.config.tuning_dir,
            self.config.tuning_id
        )
        os.makedirs(finetune_dir, exist_ok=True)
        
        # Save finetune config
        config_dict = self.config.__dict__.copy()
        # Convert device to string, avoid 'Object of type device is not JSON serializable'
        config_dict['device'] = str(config_dict['device'])
        with open(os.path.join(finetune_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        # Update finetune metadata
        metadata_path = os.path.join(self.config.experiments_dir, 'metadata/finetune.json')
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        metadata[self.config.tuning_id] = {
            'exp_id': self.config.exp_id,
            'checkpoint': self.config.checkpoint,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'config': config_dict
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Run training epochs
        start_time = time.time()
        all_metrics = []
        
        print(f"Finetuning {self.config.finetune_epoch} epochs")
        for epoch in range(self.config.finetune_epoch):
            logging.info(f"Finetuning epoch {epoch}")
            epoch_metrics = self.run_epoch(epoch)

            all_metrics.append(epoch_metrics)
            torch.save({'model': self.model.state_dict(), 'config': self.config, 'quantile': epoch_metrics['quantile']}, \
                        os.path.join(finetune_dir, f'model-{epoch}.pth'))

        results_file = os.path.join(finetune_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        total_time = time.time() - start_time
        logging.info(f"Fine-tuning completed in {total_time/60:.2f} minutes")

        return all_metrics
    