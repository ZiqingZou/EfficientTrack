from random import random

import numpy as np
import time
import pandas as pd
import torch


device = torch.device('cpu')


def generate_grf(length, mean=0, variance=1, length_scale=1):
    freqs = np.fft.fftfreq(length, 0.05)
    psd = variance * np.exp(-0.5 * (2 * np.pi * freqs * length_scale) ** 2)

    random_complex = np.fft.fft(np.random.normal(size=length) + 1j * np.random.normal(size=length))
    random_complex *= np.sqrt(psd)

    random_field = np.fft.ifft(random_complex).real
    random_field += mean

    return random_field


class DataCollector:
    def __init__(self, env, controller, file_name, save_fp, noise, train_or_test):
        self.noise = noise
        self.save_fp = save_fp
        self.env = env
        self.controller = controller
        self.file_name = file_name
        self.data = np.load('../data_process/zc_interp_npy_' + train_or_test + '/' + file_name + '.npy')
        self.data_len = len(self.data[:, 0])

        # self.env.reset(self.data[0, :].tolist() + [0, 0, 0, 0])
        # if noise != 0.0:
        #     for i in range(4):
        #         noise_len = np.random.randint(1, self.data_len - 1)
        #         position = np.random.randint(1, self.data_len - noise_len)
        #         # step_noise = np.clip(np.random.normal(0, noise), -noise, noise)
        #         step_noise = noise
        #         self.data[position:position + noise_len, i] = [ele + step_noise
        #                                                        for ele in self.data[position:position + noise_len, i]]
        #
        # for i in range(4):
        #     grf = generate_grf(length=self.data_len)
        #     self.data[:, i] = self.data[:, i] + grf * noise
        self.env.reset(self.data[0, :].tolist() + [0, 0, 0, 0])

        self.idx = 0
        self.np_target = np.zeros(4)
        self.previous_target_sequence = []

    def update_target(self):
        if self.idx < self.data.shape[0] - 1:
            self.np_target = self.data[self.idx + 1, :]
        else:
            self.np_target = None

    @torch.no_grad()
    def loopy(self, policy, model):
        print('Ready to control!!')
        data_save = {"target": [], "real_target": [], "js": [], "jv": [], "time": [], "action": []}
        saved = False

        if policy is None:
            while not saved:
                self.update_target()
                if self.np_target is not None:
                    pos_dict = self.env.get_dic_pos()
                    now_pos = np.array(list(pos_dict.values()))
                    vel_dict = self.env.get_dic_vel()
                    now_vel = np.array(list(vel_dict.values()))

                    if self.noise:
                        random_noise = np.random.standard_normal(self.np_target.shape)
                        clipped_noise = np.clip(random_noise, -1, 1)
                        target_pos = self.np_target + (clipped_noise * self.noise)
                    else:
                        target_pos = self.np_target

                    action = self.controller.pid_control(target_pos, now_pos)
                    self.env.step(action.tolist())
                    print(self.idx, action)

                    data_save["target"].append(target_pos)
                    data_save["real_target"].append(self.np_target)
                    data_save["js"].append(now_pos)
                    data_save["jv"].append(now_vel)
                    data_save["time"].append(time.time())
                    data_save["action"].append(action)
                    self.idx += 1
                else:
                    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    np.save(self.save_fp + '/{}_{}.npy'.format(self.file_name, timestamp), data_save)
                    df = pd.DataFrame(data_save)
                    csv_file = self.save_fp + '/{}_{}.csv'.format(self.file_name, timestamp)
                    df.to_csv(csv_file, index=False)
                    saved = True
                    print('\nEnd!!', time.localtime())

        else:
            H = policy.horizon
            integrate_step = 1
            while not saved:
                self.update_target()
                if self.np_target is not None:
                    pos_dict = self.env.get_dic_pos()
                    now_pos = np.array(list(pos_dict.values()))
                    vel_dict = self.env.get_dic_vel()
                    now_vel = np.array(list(vel_dict.values()))

                    state = torch.from_numpy(np.concatenate((now_pos, now_vel))).float().unsqueeze(0).to(device)
                    target = torch.from_numpy(self.data[self.idx, :]).float().unsqueeze(0).to(device)
                    if self.idx == 0:
                        previous_state = state.repeat(H, 1).unsqueeze(0)
                        previous_target = target.repeat(H, 1).unsqueeze(0)
                        future_real_target = torch.from_numpy(self.data[1:H + 1, :]).float().unsqueeze(0).to(device)
                    else:
                        previous_state = torch.cat((self.previous_state[:, 1:, :], self.state.unsqueeze(1)), dim=1)
                        previous_target = torch.cat((self.previous_target[:, 1:, :], self.target.unsqueeze(1)), dim=1)
                        if self.idx + H < self.data_len:
                            add_tg = torch.from_numpy(self.data[self.idx + H, :]).float().unsqueeze(0).to(device)
                        else:
                            add_tg = torch.from_numpy(self.data[-1, :]).float().unsqueeze(0).to(device)
                        future_real_target = torch.cat((self.future_real_target[:, 1:, :], add_tg.unsqueeze(1)), dim=1)

                    self.state = state
                    self.previous_state = previous_state
                    self.target = target
                    self.previous_target = previous_target
                    self.future_real_target = future_real_target

                    target_sequence = []
                    for k in range(integrate_step):
                        next_target, _, _ = policy.control(state, previous_state, target, previous_target,
                                                           future_real_target)
                        target_sequence.append(next_target)
                        next_state_predict, _ = model.predict(state, previous_state, target, previous_target, next_target)

                        previous_state = torch.cat((previous_state[:, 1:, :], state.unsqueeze(1)), dim=1)
                        previous_target = torch.cat((previous_target[:, 1:, :], target.unsqueeze(1)), dim=1)
                        state = next_state_predict
                        target = next_target
                        if self.idx + k + H + 1 < self.data_len:
                            add_tg = torch.from_numpy(self.data[self.idx + k + H + 1, :]).float().unsqueeze(0).to(device)
                        else:
                            add_tg = torch.from_numpy(self.data[-1, :]).float().unsqueeze(0).to(device)
                        future_real_target = torch.cat((future_real_target[:, 1:, :], add_tg.unsqueeze(1)), dim=1)

                    if integrate_step == 1:
                        target_pos = target_sequence[0]
                    else:
                        if self.idx == 0:
                            target_pos = target_sequence[0]
                            for i in range(integrate_step - 1):
                                self.previous_target_sequence.append(target_sequence)
                        elif self.idx < integrate_step - 1:
                            target_pos = target_sequence[0]
                            for i in range(1, self.idx + 1):
                                target_pos = target_pos + self.previous_target_sequence[-i][i]
                            target_pos = target_pos / (self.idx + 1)
                            self.previous_target_sequence.pop(0)
                            self.previous_target_sequence.append(target_sequence)
                        else:
                            target_pos = target_sequence[0]
                            for i in range(1, integrate_step):
                                target_pos = target_pos + self.previous_target_sequence[-i][i]
                            target_pos = target_pos / integrate_step
                            self.previous_target_sequence.pop(0)
                            self.previous_target_sequence.append(target_sequence)

                    target_pos = target_pos.squeeze().detach().cpu().numpy()
                    action = self.controller.pid_control(target_pos, now_pos)
                    self.env.step(action.tolist())
                    print(self.idx, action)

                    data_save["target"].append(target_pos)
                    data_save["real_target"].append(self.np_target)
                    data_save["js"].append(now_pos)
                    data_save["jv"].append(now_vel)
                    data_save["time"].append(time.time())
                    data_save["action"].append(action)
                    self.idx += 1
                else:
                    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    np.save(self.save_fp + '/{}_{}.npy'.format(self.file_name, timestamp), data_save)
                    df = pd.DataFrame(data_save)
                    csv_file = self.save_fp + '/{}_{}.csv'.format(self.file_name, timestamp)
                    df.to_csv(csv_file, index=False)
                    saved = True
                    print('\nEnd!!', time.localtime())

