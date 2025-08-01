import torch
import numpy as np
from sympy import false

"""
Dataset need to be:
    state: (total_length, 2 * horizon + 1, state_dim): Previous H, now, and future H,
    target: (total_length, 3 * horizon, target_dim): Previous H, now, and future 2*H - 1,
        time aligned with state_sequence.),
    real_target: (total_length, 3 * horizon, target_dim): Previous H, now, and future 2*H - 1,
        time aligned with state_sequence.),
    including: now = 0  and now = t_end(repeat in the right of future target).
"""


class Buffer:
    def __init__(self, buffer_size, horizon, state_dim, target_dim, batch_size):
        self.capacity = buffer_size
        self.horizon = horizon
        self.state_dim = state_dim
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.num_push = 0

        self.previous_state = torch.zeros(buffer_size, horizon, state_dim).to(torch.device('cuda'))
        self.state = torch.zeros(buffer_size, 1, state_dim).to(torch.device('cuda'))
        self.previous_target = torch.zeros(buffer_size, horizon, target_dim).to(torch.device('cuda'))
        self.target = torch.zeros(buffer_size, 1, target_dim).to(torch.device('cuda'))
        self.future_real_target = torch.zeros(buffer_size, horizon, target_dim).to(torch.device('cuda'))

        self.end_flag = torch.zeros(buffer_size, dtype=torch.bool).to(torch.device('cuda'))

    def sample(self):
        batch_size = self.batch_size
        if self.num_push > batch_size:
            if self.num_push > self.capacity:
                index = np.random.choice(range(self.capacity - self.horizon), batch_size, replace=False)
            else:
                index = np.random.choice(range(self.num_push), batch_size, replace=False)
        else:
            index = np.random.choice(range(self.num_push), batch_size, replace=True)
        index = torch.tensor(index).to(torch.device('cuda'))

        mask = self.end_flag[index.unsqueeze(1) + torch.arange(self.horizon, device='cuda').unsqueeze(0)]
        all_zeros = torch.all(~mask, dim=1)

        while torch.any(~all_zeros):
            bad_rows = ~all_zeros
            size = bad_rows.sum().item()
            if self.num_push > batch_size:
                if self.num_push > self.capacity:
                    new_indices = np.random.choice(range(self.capacity - self.horizon), size, replace=False)
                else:
                    new_indices = np.random.choice(range(self.num_push), size, replace=False)
            else:
                new_indices = np.random.choice(range(self.num_push), size, replace=True)
            new_indices = torch.tensor(new_indices).to(torch.device('cuda'))
            index[bad_rows] = new_indices

            mask = self.end_flag[index.unsqueeze(1) + torch.arange(self.horizon, device='cuda').unsqueeze(0)]
            all_zeros = torch.all(~mask, dim=1)

        previous_state_sampled = self.previous_state[index, :, :]
        state_sampled = self.state[index, 0:1, :]
        previous_target_sampled = self.previous_target[index, :, :]
        target_sampled = self.target[index, 0:1, :]
        future_real_target_sampled = self.future_real_target[index, :, :]

        # for loc, ele in enumerate(index):
        #     for i in range(self.horizon):
        #         if self.end_flag[ele + i]:
        #             future_state[loc, i:, :] = self.state[ele + i, 0, :].repeat(self.horizon - i, 1)
        #             future_target[loc, i:, :] = self.target[ele + i, 0, :].repeat(self.horizon - i, 1)
        #             fu_future_real_target[loc, i:, :] = (
        #                 self.future_real_target[ele + i, -1, :].repeat(self.horizon - i, 1))
        #             break
        #         else:
        #             future_state[loc, i, :] = self.state[ele + i + 1, 0, :]
        #             future_target[loc, i, :] = self.target[ele + i + 1, 0, :]
        #             if i == self.horizon - 1:
        #                 fu_future_real_target[loc, :, :] = self.future_real_target[ele + self.horizon, :, :]

        future_state = torch.cat([self.previous_state[index + self.horizon, 1:, :],
                                  self.state[index + self.horizon, 0:1, :]], dim=1)
        future_target = torch.cat([self.previous_target[index + self.horizon, 1:, :],
                                  self.target[index + self.horizon, 0:1, :]], dim=1)
        fu_future_real_target = self.future_real_target[index + self.horizon, :-1, :]
        previous_real_target = torch.zeros(batch_size, self.horizon + 1, self.target_dim).to(torch.device('cuda'))

        state = torch.cat([previous_state_sampled, state_sampled, future_state], dim=1)
        target = torch.cat([previous_target_sampled, target_sampled, future_target], dim=1)
        real_target = torch.cat([previous_real_target, future_real_target_sampled, fu_future_real_target], dim=1)

        return state, target, real_target, False

    def push(self, rl_state, end_flag=False):
        loc = self.num_push % self.capacity
        idx1 = self.horizon * self.state_dim
        idx2 = (self.horizon + 1) * self.state_dim
        idx3 = (self.horizon + 1) * self.state_dim + self.horizon * self.target_dim
        idx4 = (self.horizon + 1) * self.state_dim + (self.horizon + 1) * self.target_dim

        self.previous_state[loc, :, :].copy_(torch.tensor(
            rl_state[:idx1]).view(self.horizon, self.state_dim).to(torch.device('cuda')))
        self.state[loc, 0, :].copy_(torch.tensor(rl_state[idx1:idx2]).to(torch.device('cuda')))
        self.previous_target[loc, :, :].copy_(torch.tensor(
            rl_state[idx2:idx3]).view(self.horizon, self.target_dim).to(torch.device('cuda')))
        self.target[loc, 0, :].copy_(torch.tensor(rl_state[idx3:idx4]).to(torch.device('cuda')))
        self.future_real_target[loc, :, :].copy_(torch.tensor(
            rl_state[idx4:]).view(self.horizon, self.target_dim).to(torch.device('cuda')))

        if end_flag:
            self.end_flag[loc] = True
        else:
            self.end_flag[loc] = False

        self.num_push += 1

