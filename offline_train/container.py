import torch
from torch.utils.data import DataLoader, Dataset

"""
Dataset need to be:
    state: (total_length, 2 * horizon + 1, state_dim): Previous H, now, and future H,
    target: (total_length, 3 * horizon, target_dim): Previous H, now, and future 2*H - 1,
        time aligned with state_sequence.),
    real_target: (total_length, 3 * horizon, target_dim): Previous H, now, and future 2*H - 1,
        time aligned with state_sequence.),
    including: now = 0  and now = t_end(repeat in the right of future target).
"""


class MyDataset(Dataset):
    def __init__(self, state_path, target_path, real_target_path):
        self.state = torch.load(state_path).to(torch.device('cuda'))
        self.target = torch.load(target_path).to(torch.device('cuda'))
        self.real_target = torch.load(real_target_path).to(torch.device('cuda'))
        self.total_length = self.state.shape[0]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return self.state[idx], self.target[idx], self.real_target[idx]


class Container:
    def __init__(self, state_path=None, target_path=None, real_target_path=None, batch_size=0):
        self.dataset = MyDataset(state_path, target_path, real_target_path)
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.iterator = iter(self.dataloader)
        self.num_push = 0

    def sample(self):
        try:
            state_sampled, target_sampled, real_target_sampled = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            state_sampled, target_sampled, real_target_sampled = next(self.iterator)

        epoch_end = False

        if len(state_sampled) < self.batch_size:
            epoch_end = True

        return state_sampled, target_sampled, real_target_sampled, epoch_end
