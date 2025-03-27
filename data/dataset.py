import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import copy
import heapq
import random
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import oapackage
except ImportError:
    print("OApackage is not installed, can not use CDT.")
from scipy.optimize import minimize
from torch.nn import functional as F  # noqa
from torch.utils.data import IterableDataset
from tqdm.auto import trange  # noqa

from IPython import embed

class SequenceDataset(IterableDataset):
    """
    A dataset of sequential data.

    Args:
        data (dict): Input dataset, containing trajectory IDs and sequences of observations.
        scaler (torch.Tensor): Scaling factor for reward values.
        split (str): 'train' or 'cal'
        
        Augmentation params:
        deg (int): Degree of polynomial used for Pareto frontier augmentation.
        pf_sample (bool): Whether to sample data from the Pareto frontier.
        max_rew_decrease (float): Maximum reward decrease for Pareto frontier augmentation.
        beta (float): Parameter used for cost-based augmentation.
        augment_percent (float): Percentage of data to augment.
        max_reward (float): Maximum reward value for augmentation.
        min_reward (float): Minimum reward value for augmentation.
        cost_reverse (bool): Whether to reverse the cost values.
        pf_only (bool): Whether to use only Pareto frontier data points.
        rmin (float): Minimum reward value for random augmentation.
        cost_bins (int): Number of cost bins for random augmentation.
        npb (int): Number of data points to select from each cost bin for random augmentation.
        cost_sample (bool): Whether to sample data based on cost.
        cost_transform (callable): Function used to transform cost values.
        prob (float): Probability of sampling from each trajectory start index.
        start_sampling (bool): Whether to sample from each trajectory start index.
        random_aug (float): Percentage of data to augment randomly.
        aug_rmin (float): Minimum reward value for random augmentation.
        aug_rmax (float): Maximum reward value for random augmentation.
        aug_cmin (float): Minimum cost value for random augmentation.
        aug_cmax (float): Maximum cost value for random augmentation.
        cgap (float): Cost gap for random augmentation.
        rstd (float): Standard deviation of reward values for random augmentation.
        cstd (float): Standard deviation of cost values for random augmentation.
    """

    def __init__(
        self,
        dataset: dict,
        scaler: torch.Tensor,
        split: str,
        is_normalize: bool = True,
        is_need_idx: bool = False,
        
        deg: int = 3,
        pf_sample: bool = False,
        max_rew_decrease: float = 1.0,
        beta: float = 1.0,
        augment_percent: float = 0,
        max_reward: float = 1000.0,
        min_reward: float = 5,
        cost_reverse: bool = False,
        pf_only: bool = False,
        rmin: float = 0,
        cost_bins: int = 60,
        npb: int = 5,
        cost_sample: bool = False,
        cost_transform=lambda x: 50 - x,
        prob: float = 0.4,
        start_sampling: bool = False,
        random_aug: float = 0,
        aug_rmin: float = 0,
        aug_rmax: float = 600,
        aug_cmin: float = 5,
        aug_cmax: float = 50,
        cgap: float = 5,
        rstd: float = 1,
        cstd: float = 0.2,
    ):
        self.original_data, info = process_sequence_dataset(dataset, cost_reverse)
        self.N_cal = 100
        self.pad_to = 1024
        self.scaler = scaler
        self.is_normalize = is_normalize
        self.is_need_idx = is_need_idx
        self.split = split
        self.start_sampling = start_sampling

        self.aug_data = []
        if pf_only:
            print("*" * 100)
            print("Using pareto frontier data points only!!!!!")
            print("*" * 100)
            self.dataset = select_optimal_trajectory(self.original_data, rmin, cost_bins,
                                                     npb)
        elif random_aug > 0:
            self.idx, self.aug_data = random_augmentation(
                self.original_data,
                random_aug,
                aug_rmin,
                aug_rmax,
                aug_cmin,
                aug_cmax,
                cgap,
                rstd,
                cstd,
            )
        elif augment_percent > 0:
            # sampled data and the index of its "nearest" point in the dataset
            self.idx, self.aug_data, self.pareto_frontier, self.indices = augmentation(
                self.original_data, deg, max_rew_decrease, beta, augment_percent,
                max_reward, min_reward)
        self.dataset = self.original_data + self.aug_data
        print(
            f"original data: {len(self.original_data)}, augment data: {len(self.aug_data)}, total: {len(self.dataset)}"
        )

        if cost_sample:
            self.sample_prob = compute_cost_sample_prob(self.dataset, cost_transform)
        elif pf_sample:
            self.sample_prob = compute_sample_prob(self.dataset, self.pareto_frontier, 1)
        else:
            self.sample_prob = None

        # compute every trajectories start index sampling prob:
        if start_sampling:
            self.start_idx_sample_prob = compute_start_index_sample_prob(
                dataset=self.dataset, prob=prob)

        self.state_channel = self.dataset[0]["observations"].shape[1]
        self.nt_total = self.dataset[0]["observations"].shape[0]
        self.shape = self.__prepare_sample(0).shape

    def compute_pareto_return(self, cost):
        return self.pareto_frontier(cost)

    def __len__(self):
        return len(self.dataset)

    def __prepare_sample(self, traj_idx):
        traj = self.dataset[traj_idx]
        states = traj["observations"]
        actions = traj["actions"]
        rewards = traj["rewards"]
        costs = traj["costs"]

        # time_steps = np.arange(0, states.shape[0])

        # pad up to seq_len if needed
        # mask = np.hstack(
        #     [np.ones(states.shape[0]),
        #      np.zeros(self.pad_to - states.shape[0])])

        states = torch.tensor(pad_along_axis(states, pad_to=self.pad_to).reshape(self.pad_to, -1)) # 1024, 72
        actions = torch.tensor(pad_along_axis(actions, pad_to=self.pad_to).reshape(self.pad_to, -1)) # 1024, 2
        rewards = torch.tensor(pad_along_axis(rewards, pad_to=self.pad_to).reshape(self.pad_to, -1)) # 1024, 1
        costs = torch.tensor(pad_along_axis(costs, pad_to=self.pad_to).reshape(self.pad_to, -1)) # 1024, 1

        if self.is_normalize:
            returns = torch.cat((states, actions, rewards, costs), dim=1).permute(1, 0) / self.scaler
        else:
            returns = torch.cat((states, actions, rewards, costs), dim=1).permute(1, 0)
            
        if self.is_need_idx:
            return returns, traj_idx
        else:
            return returns


    def __iter__(self):
        while True: 
            if self.split == 'train':
                traj_idx = np.random.choice(len(self.dataset)-self.N_cal, p=self.sample_prob)
            elif self.split == 'cal':
                traj_idx = np.random.choice(range(len(self.dataset)-self.N_cal, len(self.dataset)), p=self.sample_prob)
            else:
                raise ValueError("split must be one of ['train', 'cal']")
            yield self.__prepare_sample(traj_idx)
            

def pad_along_axis(arr: np.ndarray,
                   pad_to: int,
                   axis: int = 0,
                   fill_value: float = 0.0) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Calculate the discounted cumulative sum of x (can be rewards or costs).
    """
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def process_sequence_dataset(dataset: dict, cost_reverse: bool = False):
    '''
    Processe a given dataset into a list of trajectories, each containing information about 
    the observations, actions, rewards, costs, returns, and cost returns for a single episode.
    
    Args:

        dataset (dict): A dictionary representing the dataset, 
                        with keys "observations", "actions", "rewards", "costs", "terminals", and "timeouts", 
                        each containing numpy arrays of corresponding data.
        cost_reverse (bool): An optional boolean parameter that indicates whether the cost should be reversed.
        
    Returns:
        traj (list): A list of dictionaries, each representing a trajectory.
        info (dict): A dictionary containing additional information about the trajectories

    '''
    traj, traj_len = [], []
    data_, episode_step = defaultdict(list), 0
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])
        if cost_reverse:
            data_["costs"].append(1.0 - dataset["costs"][i])
        else:
            data_["costs"].append(dataset["costs"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(episode_data["rewards"], gamma=1)
            episode_data["cost_returns"] = discounted_cumsum(episode_data["costs"],
                                                             gamma=1)
            traj.append(episode_data)
            traj_len.append(episode_step)
            # reset trajectory buffer
            data_, episode_step = defaultdict(list), 0
        episode_step += 1

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info

if __name__ == "__main__":
    from dsrl.infos import DENSITY_CFG
    from configs.pretrain_config import TrainConfig, get_train_config, TASK_CONFIG

    import bullet_safety_gym  # noqa
    import dsrl
    import gymnasium as gym  # noqa

    config = get_train_config(exp_id="turbo_MultiScaler", model_size="turbo")
    config = TASK_CONFIG[config.task]()
    config.scaler = torch.tensor(config.scaler).reshape(-1, 1)
    
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
        split='train',
        is_normalize=False
    )

    dataloader = DataLoader(dataset, batch_size = 2000, pin_memory = True, num_workers = 10)
    scaler = torch.zeros(dataset.shape[0])
    N_iter = 40
    for i, data in enumerate(dataloader):
        # print(data.shape)
        scaler = np.maximum(scaler, data.abs().max(2)[0].max(0)[0])
        print(scaler)
        if i == N_iter:
            break
    print(scaler)
    print(dataset.shape, dataset.state_channel, dataset.nt_total)

# class TokamakDataset(Dataset):
#     def __init__(self, split='train', is_normalize=True, is_need_idx=False):
#         dataset = load_from_disk(r"/data/hupeiyan/tokamak_data/consolidated_dataset")
#         dataset.set_format('torch')
        
#         # Split dataset based on mode
#         if split == 'train':
#             self.dataset = dataset.select(range(48950))
#         elif split == 'cal':
#             self.dataset = dataset.select(range(48950, 49950))
#         elif split == 'test':
#             self.dataset = dataset.select(range(49950, 50000))
#         else:
#             raise ValueError("split must be one of ['train', 'cal', 'test']")
            
#         self.features = dataset.features
#         self.pad_size = 128
#         self.nt_total = 122

#         # CHOICE: use single scaler or multiple scaler
#         # self.scaler = 10.0
#         self.scaler = torch.tensor([2, 7, 2, 1, 2, 2, 2, 2, 1, 1, 2, 3]).reshape(12,1)

#         self.is_normalize = is_normalize
#         self.is_need_idx = is_need_idx

#     def __len__(self):
#         return len(self.dataset)

#     def _process_data(self, data):
#         """Process data to fit model input"""
#         states = data['outputs'][:, [1, 4, 6]].permute(1, 0)  # Convert to [3, 122]
#         actions = data['actions'].permute(1, 0)  # Convert to [9, 121]

#         nt = states.shape[1]
#         states = torch.nn.functional.pad(states, (0, self.pad_size - nt, 0, 0), 'constant', 0)
#         actions = torch.nn.functional.pad(actions, (0, self.pad_size - nt + 1, 0, 0), 'constant', 0)  # padding to [9, 128]
#         data = torch.cat((states, actions), dim=0)  # Stack to [12, 128]

#         if self.is_normalize:
#             data = data / self.scaler
        
#         return data

#     def __getitem__(self, idx):
#         """Get a single data item"""
#         data = self.dataset[idx]

#         if self.is_need_idx:
#             return self._process_data(data), idx
#         else:
#             return self._process_data(data)

# if __name__ == "__main__":
#     from IPython import embed
#     from tqdm import tqdm
#     device = 'cuda:0'
#     safe_bound = 4.98
#     dataset = TokamakDataset(split='train')
#     dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=10)
    
#     q95_mean = 0
#     below_samples = 0
#     below_points = 0
#     total_samples = 0
#     safe_samples = 0
#     for i, data in tqdm(enumerate(dataloader)):
#         data = data.to(device)
#         data = data * dataset.scaler.unsqueeze(0).to(device)
#         q95 = data[:, 1, :dataset.nt_total]  # Get q95 values
        
#         # Calculate mean
#         q95_mean += q95.mean().item() * len(data)
#         total_samples += len(data)
        
#         # Calculate below samples and points
#         below_mask = q95 < safe_bound
#         below_samples += (below_mask.any(dim=-1)).sum().item()
#         below_points += below_mask.sum().item()
        
#         # Calculate samples that are always above safe_bound
#         safe_samples += ((q95 >= safe_bound).all(dim=-1)).sum().item()

#     q95_mean = q95_mean / total_samples
#     print('train:')
#     print(f"Q95 mean: {q95_mean:.4f}")
#     print(f"Percentage of samples below {safe_bound}: {below_samples / total_samples:.4f}")
#     print(f"Percentage of time points below {safe_bound}: {below_points / (total_samples * dataset.nt_total):.4f}")
#     print(f"Percentage of samples always above {safe_bound}: {safe_samples / total_samples:.4f}")

#     dataset = TokamakDataset(split='test')
#     dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
#     q95_mean = 0
#     below_samples = 0
#     below_init_points = 0
#     below_points = 0
#     total_samples = 0
#     for i, data in tqdm(enumerate(dataloader)):
#         data = data.to(device)
#         data = data * dataset.scaler.unsqueeze(0).to(device)
#         q95 = data[:, 1, :dataset.nt_total]  # Get q95 values
#         q95_mean += q95.mean().item() * len(data)
#         total_samples += len(data)
#         below_mask = q95 < safe_bound
#         below_samples += (below_mask.any(dim=-1)).sum().item()
#         below_init_points += below_mask[:, 0].sum().item()
#         below_points += below_mask.sum().item()
#     q95_mean = q95_mean / total_samples
#     print('test:')
#     print(f"Q95 mean: {q95_mean:.4f}")
#     print(f"Percentage of samples below {safe_bound}: {below_samples / total_samples:.4f}")
#     print(f"Percentage of time points below {safe_bound}: {below_points / (total_samples * dataset.nt_total):.4f}")
#     print(f"Percentage of initial time points below {safe_bound}: {below_init_points / (total_samples):.4f}")