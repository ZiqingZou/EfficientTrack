import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import shutil

from network_model.predictor import Predictor
from network_model.parser import parse_config

cfg = parse_config('../config.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
H = 20  # horizon

model = Predictor(cfg)
# model.load('../offline_train/state_dict-82/4992_predictor_193_total4992.pth')
# model.load('../offline_train/state_dict/106610_predictor_70_1523.pth')
model.load('../offline_train/state_dict/760000_predictor_2000_380.pth')

folder_path = '../origin_noise_data/data_without_noise_test'
folder_path = os.path.abspath(folder_path)

save_path1 = '../simulation/figure_model_test'
shutil.rmtree(save_path1, ignore_errors=True)
os.makedirs(save_path1, exist_ok=True)

save_path2 = '../simulation/outcome_data'
os.makedirs(save_path2, exist_ok=True)
file_name2 = 'model_test_metrics.csv'


def model_multi_step_plot():

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            pos_series = np.array(data['js'])
            vel_series = np.array(data['jv'])
            target_series = np.array(data['target'])

            prediction = []
            for step_pos in range(0, len(data['time']), H):
                # initialize
                pos_now = pos_series[step_pos, :]
                vel_now = vel_series[step_pos, :]
                state_now = np.hstack((pos_now, vel_now))
                state = torch.from_numpy(state_now).float().unsqueeze(0).to(device)

                next_target = torch.from_numpy(target_series[step_pos, :]).float().unsqueeze(0).to(device)

                if step_pos == 0:
                    target = next_target
                else:
                    target = torch.from_numpy(target_series[step_pos - 1, :]).float().unsqueeze(0).to(device)

                previous_state = torch.zeros(1, H, state.shape[1]).to(device)
                previous_target = torch.zeros(1, H, next_target.shape[1]).to(device)
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

                # multi step predict
                for i in range(H):
                    next_state_predict, _ = model.predict(state, previous_state, target, previous_target, next_target)
                    prediction.append(next_state_predict.squeeze(0).detach().cpu().numpy())

                    previous_state = torch.cat((previous_state[:, 1:, :], state.unsqueeze(1)), dim=1)
                    previous_target = torch.cat((previous_target[:, 1:, :], target.unsqueeze(1)), dim=1)
                    state = next_state_predict
                    target = next_target
                    if step_pos + i + 1 < len(data['time']):
                        next_target = torch.from_numpy(target_series[step_pos + i + 1, :]).float().unsqueeze(0).to(device)
                    else:
                        next_target = torch.from_numpy(target_series[-1, :]).float().unsqueeze(0).to(device)

            # plot
            t_state = range(len(data['time']))
            t_predict = range(1, len(prediction) + 1)
            prediction = np.array(prediction)
            fig, axs = plt.subplots(2, 4, figsize=(24, 8))

            axs[0, 0].plot(t_state, pos_series[:, 0], label='pos boom', color='#9DA8AF', linewidth=2, linestyle='--')
            for i in range(0, len(t_predict), H):
                axs[0, 0].plot(t_predict[i:i + H], prediction[i:i + H, 0], color='#2878B5', linewidth=2)
            axs[0, 0].plot([], [], color='#2878B5', label='predict')
            axs[0, 0].set_xlabel('step', fontsize=18)
            axs[0, 0].set_ylabel('pos angle (rad)', fontsize=18)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 0].legend(fontsize=15)

            axs[0, 1].plot(t_state, pos_series[:, 1], label='pos arm', color='#9DA8AF', linewidth=2, linestyle='--')
            for i in range(0, len(t_predict), H):
                axs[0, 1].plot(t_predict[i:i + H], prediction[i:i + H, 1], color='#2878B5', linewidth=2)
            axs[0, 1].plot([], [], color='#2878B5', label='predict')
            axs[0, 1].set_xlabel('step', fontsize=18)
            axs[0, 1].set_ylabel('pos angle (rad)', fontsize=18)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 1].legend(fontsize=15)

            axs[0, 2].plot(t_state, pos_series[:, 2], label='pos swing', color='#9DA8AF', linewidth=2, linestyle='--')
            for i in range(0, len(t_predict), H):
                axs[0, 2].plot(t_predict[i:i + H], prediction[i:i + H, 2], color='#2878B5', linewidth=2)
            axs[0, 2].plot([], [], color='#2878B5', label='predict')
            axs[0, 2].set_xlabel('step', fontsize=18)
            axs[0, 2].set_ylabel('pos angle (rad)', fontsize=18)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 2].legend(fontsize=15)

            axs[0, 3].plot(t_state, pos_series[:, 3], label='pos bucket', color='#9DA8AF', linewidth=2, linestyle='--')
            for i in range(0, len(t_predict), H):
                axs[0, 3].plot(t_predict[i:i + H], prediction[i:i + H, 3], color='#2878B5', linewidth=2)
            axs[0, 3].plot([], [], color='#2878B5', label='predict')
            axs[0, 3].set_xlabel('step', fontsize=18)
            axs[0, 3].set_ylabel('pos angle (rad)', fontsize=18)
            axs[0, 3].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 3].legend(fontsize=15)

            axs[1, 0].plot(t_state, vel_series[:, 0], label='vel boom', color='#9DA8AF', linewidth=2, linestyle='--')
            for i in range(0, len(t_predict), H):
                axs[1, 0].plot(t_predict[i:i + H], prediction[i:i + H, 4], color='#2878B5', linewidth=2)
            axs[1, 0].plot([], [], color='#2878B5', label='predict')
            axs[1, 0].set_xlabel('step', fontsize=18)
            axs[1, 0].set_ylabel('error (rad/s)', fontsize=18)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 0].legend(fontsize=15)

            axs[1, 1].plot(t_state, vel_series[:, 1], label='vel arm', color='#9DA8AF', linewidth=2, linestyle='--')
            for i in range(0, len(t_predict), H):
                axs[1, 1].plot(t_predict[i:i + H], prediction[i:i + H, 5], color='#2878B5', linewidth=2)
            axs[1, 1].plot([], [], color='#2878B5', label='predict')
            axs[1, 1].set_xlabel('step', fontsize=18)
            axs[1, 1].set_ylabel('error (rad/s)', fontsize=18)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 1].legend(fontsize=15)

            axs[1, 2].plot(t_state, vel_series[:, 2], label='vel swing', color='#9DA8AF', linewidth=2, linestyle='--')
            for i in range(0, len(t_predict), H):
                axs[1, 2].plot(t_predict[i:i + H], prediction[i:i + H, 6], color='#2878B5', linewidth=2)
            axs[1, 2].plot([], [], color='#2878B5', label='predict')
            axs[1, 2].set_xlabel('step', fontsize=18)
            axs[1, 2].set_ylabel('error (rad/s)', fontsize=18)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 2].legend(fontsize=15)

            axs[1, 3].plot(t_state, vel_series[:, 3], label='vel bucket', color='#9DA8AF', linewidth=2, linestyle='--')
            for i in range(0, len(t_predict), H):
                axs[1, 3].plot(t_predict[i:i + H], prediction[i:i + H, 7], color='#2878B5', linewidth=2)
            axs[1, 3].plot([], [], color='#2878B5', label='predict')
            axs[1, 3].set_xlabel('step', fontsize=18)
            axs[1, 3].set_ylabel('error (rad/s)', fontsize=18)
            axs[1, 3].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 3].legend(fontsize=15)

            plt.tight_layout()
            plt.savefig(save_path1 + '/intrep_target_' + filename.replace('.npy', '') + '.png')
            plt.close(fig)


def model_multi_step_analysis():

    file_mae = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            pos_series = np.array(data['js'])
            vel_series = np.array(data['jv'])
            target_series = np.array(data['target'])

            prediction = []
            for step_pos in range(0, len(data['time']) - H):
                # initialize
                pos_now = pos_series[step_pos, :]
                vel_now = vel_series[step_pos, :]
                state_now = np.hstack((pos_now, vel_now))
                state = torch.from_numpy(state_now).float().unsqueeze(0).to(device)

                next_target = torch.from_numpy(target_series[step_pos, :]).float().unsqueeze(0).to(device)

                if step_pos == 0:
                    target = next_target
                else:
                    target = torch.from_numpy(target_series[step_pos - 1, :]).float().unsqueeze(0).to(device)

                previous_state = torch.zeros(1, H, state.shape[1]).to(device)
                previous_target = torch.zeros(1, H, next_target.shape[1]).to(device)
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

                # multi step predict
                for i in range(H):
                    next_state_predict, _ = model.predict(state, previous_state, target, previous_target, next_target)
                    prediction.append(next_state_predict.squeeze(0).detach().cpu().numpy())

                    previous_state = torch.cat((previous_state[:, 1:, :], state.unsqueeze(1)), dim=1)
                    previous_target = torch.cat((previous_target[:, 1:, :], target.unsqueeze(1)), dim=1)
                    state = next_state_predict
                    target = next_target
                    if step_pos + i + 1 < len(data['time']):
                        next_target = torch.from_numpy(target_series[step_pos + i + 1, :]).float().unsqueeze(0).to(
                            device)
                    else:
                        next_target = torch.from_numpy(target_series[-1, :]).float().unsqueeze(0).to(device)

            prediction = np.array(prediction)
            total_len = len(prediction)
            single_len = len(pos_series)

            pos_mae = []
            vel_mae = []
            for i in range(H):
                predict_complex = np.exp(1j * prediction[i:total_len:H, :])
                pos_complex = np.exp(1j * pos_series[i + 1:single_len - H + i + 1, :])
                abs_pos_error = np.abs(np.angle(pos_complex / predict_complex[:, :4]))
                pos_mae.append(np.mean(abs_pos_error, axis=0))

                abs_vel_error = np.abs(vel_series[i + 1:single_len - H + i + 1, :] - prediction[i:total_len:H, 4:])
                vel_mae.append(np.mean(abs_vel_error, axis=0))

            file_mae.append(np.hstack((np.array(pos_mae), np.array(vel_mae))))

    total_mae = np.mean(np.stack(file_mae), axis=0)
    step_total_mae = np.mean(total_mae, axis=0)

    result = np.vstack((total_mae, step_total_mae))

    df = pd.DataFrame(result, columns=['boom_pos', 'arm_pos', 'swing_pos', 'bucket_pos',
                                       'boom_vel', 'arm_vel', 'swing_vel', 'bucket_vel'])
    df = df.rename(index=lambda x: 'step_' + str(x + 1))
    df = df.rename(index={df.index[-1]: 'step mean'})
    df.to_csv(save_path2 + '/' + file_name2)


if __name__ == '__main__':
    model_multi_step_plot()
    model_multi_step_analysis()

