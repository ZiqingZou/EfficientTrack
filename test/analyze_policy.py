import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import shutil

from network_model.predictor import Predictor
from network_model.controller import Controller
from network_model.parser import parse_config

cfg = parse_config('../config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H = 20  # horizon

model = Predictor(cfg)
model.load('../offline_train/state_dict/16000_predictor_1000_16.pth')
policy = Controller(cfg)
policy.load('../offline_train/state_dict/16000_controller_1000_16.pth')

folder_path = '../origin_noise_data/data_without_noise_train'
folder_path = os.path.abspath(folder_path)

save_path1 = '../simulation/figure_policy_train'
shutil.rmtree(save_path1, ignore_errors=True)
os.makedirs(save_path1)

save_path1 = '../simulation/figure_policy_train'
shutil.rmtree(save_path1, ignore_errors=True)
os.makedirs(save_path1)

save_path2 = '../simulation/outcome_data'
file_name2 = 'policy_train_metrics.csv'


def policy_multi_step_plot():

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            pos_series = np.array(data['js'])
            vel_series = np.array(data['jv'])
            target_series = np.array(data['target'])
            data_length = len(data['time'])

            prediction = []
            for step_pos in range(0, data_length, H):
                # initialize
                pos_now = pos_series[step_pos, :]
                vel_now = vel_series[step_pos, :]
                state_now = np.hstack((pos_now, vel_now))
                state = torch.from_numpy(state_now).float().unsqueeze(0).to(device)

                if step_pos == 0:
                    target = torch.from_numpy(target_series[0, :]).float().unsqueeze(0).to(device)
                else:
                    target = torch.from_numpy(target_series[step_pos - 1, :]).float().unsqueeze(0).to(device)

                previous_state = torch.zeros(1, H, state.shape[1]).to(device)
                previous_target = torch.zeros(1, H, target.shape[1]).to(device)
                for i in range(H):
                    if step_pos - i - 1 >= 0:
                        ps = np.hstack((pos_series[step_pos - i - 1, :], vel_series[step_pos - i - 1, :]))
                        previous_state[0, H - i - 1, :] = torch.tensor(ps).to(device)
                    else:
                        ps = np.hstack((pos_series[0, :], vel_series[0, :]))
                        previous_state[0, H - i - 1, :] = torch.tensor(ps).to(device)
                    if step_pos - i - 2 >= 0:
                        previous_target[0, H - i - 1, :] = torch.tensor(target_series[step_pos - i - 2, :]).to(device)
                    else:
                        previous_target[0, H - i - 1, :] = torch.tensor(target_series[0, :]).to(device)

                future_real_target = torch.zeros(1, H, target.shape[1]).to(device)
                for i in range(H):
                    if step_pos + i < data_length:
                        future_real_target[0, i, :] = torch.tensor(target_series[step_pos + i, :]).to(device)
                    else:
                        future_real_target[0, i, :] = torch.tensor(target_series[data_length - 1, :]).to(device)

                # multi step predict
                for i in range(H):
                    next_target, _, _ = policy.control(state, previous_state, target, previous_target, future_real_target)
                    next_state_predict, _ = model.predict(state, previous_state, target, previous_target, next_target)
                    prediction.append(next_state_predict.squeeze(0).detach().cpu().numpy())

                    previous_state = torch.cat((previous_state[:, 1:, :], state.unsqueeze(1)), dim=1)
                    previous_target = torch.cat((previous_target[:, 1:, :], target.unsqueeze(1)), dim=1)
                    state = next_state_predict
                    target = next_target
                    if step_pos + i + H < data_length:
                        add_tg = torch.from_numpy(target_series[step_pos + i + H, :]).float().unsqueeze(0).to(device)
                    else:
                        add_tg = torch.from_numpy(target_series[-1, :]).float().unsqueeze(0).to(device)
                    future_real_target = torch.cat((future_real_target[:, 1:, :], add_tg.unsqueeze(1)), dim=1)

            # plot
            t_target = range(1, data_length + 1)
            t_predict = range(1, len(prediction) + 1)
            prediction = np.array(prediction)
            fig, axs = plt.subplots(1, 4, figsize=(30, 5))

            for i in range(0, len(t_predict), H):
                axs[0].plot(t_predict[i:i + H], prediction[i:i + H, 0], color='b')
            axs[0].plot([], [], color='b', label='tracking')
            axs[0].plot(t_target, target_series[:, 0], label='target boom', color='r')
            axs[0].set_xlabel('step')
            axs[0].set_ylabel('pos angle (rad)')
            axs[0].legend()

            for i in range(0, len(t_predict), H):
                axs[1].plot(t_predict[i:i + H], prediction[i:i + H, 1], color='b')
            axs[1].plot([], [], color='b', label='tracking')
            axs[1].plot(t_target, target_series[:, 1], label='target arm', color='r')
            axs[1].set_xlabel('step')
            axs[1].set_ylabel('pos angle (rad)')
            axs[1].legend()

            for i in range(0, len(t_predict), H):
                axs[2].plot(t_predict[i:i + H], prediction[i:i + H, 2], color='b')
            axs[2].plot([], [], color='b', label='tracking')
            axs[2].plot(t_target, target_series[:, 2], label='target swing', color='r')
            axs[2].set_xlabel('step')
            axs[2].set_ylabel('pos angle (rad)')
            axs[2].legend()

            for i in range(0, len(t_predict), H):
                axs[3].plot(t_predict[i:i + H], prediction[i:i + H, 3], color='b')
            axs[3].plot([], [], color='b', label='tracking')
            axs[3].plot(t_target, target_series[:, 3], label='pos bucket', color='r')
            axs[3].set_xlabel('step')
            axs[3].set_ylabel('pos angle (rad)')
            axs[3].legend()

            plt.tight_layout()
            plt.savefig(save_path1 + '/intrep_target_' + filename.replace('.npy', '') + '.png')
            plt.close(fig)


def policy_multi_step_analysis():

    file_mae = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            pos_series = np.array(data['js'])
            vel_series = np.array(data['jv'])
            target_series = np.array(data['target'])
            data_length = len(data['time'])

            prediction = []
            for step_pos in range(0, data_length - H + 1):
                # initialize
                pos_now = pos_series[step_pos, :]
                vel_now = vel_series[step_pos, :]
                state_now = np.hstack((pos_now, vel_now))
                state = torch.from_numpy(state_now).float().unsqueeze(0).to(device)

                if step_pos == 0:
                    target = torch.from_numpy(target_series[0, :]).float().unsqueeze(0).to(device)
                else:
                    target = torch.from_numpy(target_series[step_pos - 1, :]).float().unsqueeze(0).to(device)

                previous_state = torch.zeros(1, H, state.shape[1]).to(device)
                previous_target = torch.zeros(1, H, target.shape[1]).to(device)
                for i in range(H):
                    if step_pos - i - 1 >= 0:
                        ps = np.hstack((pos_series[step_pos - i - 1, :], vel_series[step_pos - i - 1, :]))
                        previous_state[0, H - i - 1, :] = torch.tensor(ps).to(device)
                    else:
                        ps = np.hstack((pos_series[0, :], vel_series[0, :]))
                        previous_state[0, H - i - 1, :] = torch.tensor(ps).to(device)
                    if step_pos - i - 2 >= 0:
                        previous_target[0, H - i - 1, :] = torch.tensor(target_series[step_pos - i - 2, :]).to(device)
                    else:
                        previous_target[0, H - i - 1, :] = torch.tensor(target_series[0, :]).to(device)

                future_real_target = torch.zeros(1, H, target.shape[1]).to(device)
                for i in range(H):
                    if step_pos + i < data_length:
                        future_real_target[0, i, :] = torch.tensor(target_series[step_pos + i, :]).to(device)
                    else:
                        future_real_target[0, i, :] = torch.tensor(target_series[data_length - 1, :]).to(device)

                # multi step predict
                for i in range(H):
                    next_target, _, _ = policy.control(state, previous_state, target, previous_target, future_real_target)
                    next_state_predict, _ = model.predict(state, previous_state, target, previous_target, next_target)
                    prediction.append(next_state_predict.squeeze(0).detach().cpu().numpy())

                    previous_state = torch.cat((previous_state[:, 1:, :], state.unsqueeze(1)), dim=1)
                    previous_target = torch.cat((previous_target[:, 1:, :], target.unsqueeze(1)), dim=1)
                    state = next_state_predict
                    target = next_target
                    if step_pos + i + H < data_length:
                        add_tg = torch.from_numpy(target_series[step_pos + i + H, :]).float().unsqueeze(0).to(device)
                    else:
                        add_tg = torch.from_numpy(target_series[-1, :]).float().unsqueeze(0).to(device)
                    future_real_target = torch.cat((future_real_target[:, 1:, :], add_tg.unsqueeze(1)), dim=1)

            prediction = np.array(prediction)
            total_len = len(prediction)
            single_len = len(target_series)

            pos_mae = []
            for i in range(H):
                predict_complex = np.exp(1j * prediction[i:total_len:H, :])
                target_complex = np.exp(1j * target_series[i:single_len - H + i + 1, :])
                abs_pos_error = np.abs(np.angle(target_complex / predict_complex[:, :4]))
                pos_mae.append(np.mean(abs_pos_error, axis=0))

            file_mae.append(np.array(pos_mae))

    total_mae = np.mean(np.stack(file_mae), axis=0)
    step_total_mae = np.mean(total_mae, axis=0)

    result = np.vstack((total_mae, step_total_mae))

    df = pd.DataFrame(result, columns=['boom_pos', 'arm_pos', 'swing_pos', 'bucket_pos'])
    df = df.rename(index=lambda x: 'step_' + str(x + 1))
    df = df.rename(index={df.index[-1]: 'step mean'})
    df.to_csv(save_path2 + '/' + file_name2)


if __name__ == '__main__':
    policy_multi_step_plot()
    policy_multi_step_analysis()

