import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from pos_to_xyz_transform import get_arm_end_xyz

namee = 'SAC_Open'
folder_path = os.path.dirname(__file__) + '/' + namee + '/test_data_test'
folder_path = os.path.abspath(folder_path)

save_fp = os.path.dirname(__file__) + '/' + namee + '/figure_test_test'
shutil.rmtree(save_fp, ignore_errors=True)
os.makedirs(save_fp, exist_ok=True)

save_path2 = os.path.dirname(__file__) + '/' + namee + '/outcome_data/test_test'
os.makedirs(os.path.dirname(__file__) + '/' + namee + '/outcome_data', exist_ok=True)


def pid_plot():

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            pos_series = np.array(data['js'])
            target_series = np.array(data['target'])
            real_target_series = np.array(data['real_target'])

            t_pos = range(len(data['time']))
            t_target = range(1, len(data['time']))

            fig, axs = plt.subplots(2, 4, figsize=(24, 8))

            axs[0, 0].plot(t_target, target_series[:-1, 0], label='policy_target', color='#2878B5')
            axs[0, 0].plot(t_target, real_target_series[:-1, 0], label='real_target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 0].plot(t_pos, pos_series[:, 0], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 0].set_xlabel('step', fontsize=18)
            axs[0, 0].set_ylabel('boom angle (rad)', fontsize=18)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 0].legend(fontsize=15)

            axs[0, 1].plot(t_target, target_series[:-1, 1], label='policy_target', color='#2878B5')
            axs[0, 1].plot(t_target, real_target_series[:-1, 1], label='real_target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 1].plot(t_pos, pos_series[:, 1], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 1].set_xlabel('step', fontsize=18)
            axs[0, 1].set_ylabel('arm angle (rad)', fontsize=18)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 1].legend(fontsize=15)

            axs[0, 2].plot(t_target, target_series[:-1, 2], label='policy_target', color='#2878B5')
            axs[0, 2].plot(t_target, real_target_series[:-1, 2], label='real_target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 2].plot(t_pos, pos_series[:, 2], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 2].set_xlabel('step', fontsize=18)
            axs[0, 2].set_ylabel('swing angle (rad)', fontsize=18)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 2].legend(fontsize=15)

            axs[0, 3].plot(t_target, target_series[:-1, 3], label='policy_target', color='#2878B5')
            axs[0, 3].plot(t_target, real_target_series[:-1, 3], label='real_target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 3].plot(t_pos, pos_series[:, 3], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 3].set_xlabel('step', fontsize=18)
            axs[0, 3].set_ylabel('bucket angle (rad)', fontsize=18)
            axs[0, 3].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 3].legend(fontsize=15)

            axs[1, 0].plot(t_target, pos_series[1:, 0] - real_target_series[:-1, 0], label='boom_error', color='#CC5F5A')
            axs[1, 0].set_xlabel('step', fontsize=18)
            axs[1, 0].set_ylabel('error angle (rad)', fontsize=18)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 0].legend(fontsize=15)

            axs[1, 1].plot(t_target, pos_series[1:, 1] - real_target_series[:-1, 1], label='arm_error', color='#CC5F5A')
            axs[1, 1].set_xlabel('step', fontsize=18)
            axs[1, 1].set_ylabel('error angle (rad)', fontsize=18)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 1].legend(fontsize=15)

            axs[1, 2].plot(t_target, pos_series[1:, 2] - real_target_series[:-1, 2], label='swing_error', color='#CC5F5A')
            axs[1, 2].set_xlabel('step', fontsize=18)
            axs[1, 2].set_ylabel('error angle (rad)', fontsize=18)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 2].legend(fontsize=15)

            axs[1, 3].plot(t_target, pos_series[1:, 3] - real_target_series[:-1, 3], label='bucket_error', color='#CC5F5A')
            axs[1, 3].set_xlabel('step', fontsize=18)
            axs[1, 3].set_ylabel('error angle (rad)', fontsize=18)
            axs[1, 3].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 3].legend(fontsize=15)

            plt.tight_layout()
            plt.savefig(save_fp + '/intrep_target_' + filename.replace('.npy', '') + '.png')
            plt.close(fig)


def compute_end_dis(pos_series, target_series):
    dis = np.zeros(len(pos_series))
    for i in range(len(pos_series)):
        pos_xyz = get_arm_end_xyz(pos_series[i, :])
        target_xyz = get_arm_end_xyz(target_series[i, :])
        dis[i] = np.linalg.norm(pos_xyz - target_xyz)
    return dis


def pid_analysis():

    file_mae = []
    file_maxe = []
    file_msre = []

    file_1dsmooth = []
    file_2dsmooth = []

    file_last_ae = []
    file_me = []

    file_mean_end = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            pos_series = np.array(data['js'])
            real_target_series = np.array(data['real_target'])

            pos_complex = np.exp(1j * pos_series[1:, :])
            real_target_complex = np.exp(1j * real_target_series[:-1, :])
            abs_error = np.abs(np.angle(pos_complex / real_target_complex))
            square_error = np.square(abs_error)

            smooth_1d = np.square(pos_series[1:, :] - pos_series[:-1, :])
            smooth_2d = np.square(pos_series[2:, :] - 2 * pos_series[1:-1, :] + pos_series[:-2, :])

            mean_abs_error = np.mean(abs_error, axis=0)
            file_mae.append(mean_abs_error)
            mean_square_root_error = np.sqrt(np.mean(square_error, axis=0))
            file_msre.append(mean_square_root_error)
            max_abs_error = np.max(abs_error, axis=0)
            file_maxe.append(max_abs_error)

            mean_square_root_1dsmooth = np.sqrt(np.mean(smooth_1d, axis=0))
            file_1dsmooth.append(mean_square_root_1dsmooth)
            mean_square_root_2dsmooth = np.sqrt(np.mean(smooth_2d, axis=0))
            file_2dsmooth.append(mean_square_root_2dsmooth)

            file_last_ae.append(np.abs(pos_series[-1, :] - real_target_series[-2, :]))
            file_me.append(np.mean(np.angle(pos_complex / real_target_complex), axis=0))

            end_dis = compute_end_dis(pos_series[1:, :], real_target_series[:-1, :])  # m
            mean_end = np.mean(end_dis)
            file_mean_end.append(mean_end)
    file_mae = np.array(file_mae)
    file_maxe = np.array(file_maxe)
    file_msre = np.array(file_msre)
    file_1dsmooth = np.array(file_1dsmooth)
    file_2dsmooth = np.array(file_2dsmooth)
    file_last_ae = np.array(file_last_ae)
    file_me = np.array(file_me)
    file_mean_end = np.array(file_mean_end).reshape((-1, 1))

    total_mae = np.mean(file_mae, axis=0) * 180 / np.pi
    total_maxe = np.mean(file_maxe, axis=0) * 180 / np.pi
    total_msre = np.mean(file_msre, axis=0) * 180 / np.pi
    total_1dsmooth = np.mean(file_1dsmooth, axis=0) * 180 / np.pi
    total_2dsmooth = np.mean(file_2dsmooth, axis=0) * 180 / np.pi
    total_last_mae = np.mean(file_last_ae, axis=0) * 180 / np.pi
    total_me = np.mean(file_me, axis=0) * 180 / np.pi
    total_mean_end = np.mean(file_mean_end)  # m

    all_mae = np.mean(total_mae)
    all_msre = np.mean(total_msre)
    all_1dsmooth = np.mean(total_1dsmooth)
    all_2dsmooth = np.mean(total_2dsmooth)
    all_me = np.mean(total_me)
    all_last_mae = np.mean(total_last_mae)

    top = np.hstack((file_mae, file_maxe, file_msre,
                     file_1dsmooth, file_2dsmooth, file_last_ae, file_me, file_mean_end))
    bottom = np.hstack((total_mae, total_maxe, total_msre,
                        total_1dsmooth, total_2dsmooth, total_last_mae, total_me, total_mean_end))
    result = np.vstack((top, bottom))

    df = pd.DataFrame(result, columns=['boom_mae', 'arm_mae', 'swing_mae', 'bucket_mae',
                                       'boom_maxe', 'arm_maxe', 'swing_maxe', 'bucket_maxe',
                                       'boom_msre', 'arm_msre', 'swing_msre', 'bucket_msre',
                                       'boom_1dsmooth', 'arm_1dsmooth', 'swing_1dsmooth', 'bucket_1dsmooth',
                                       'boom_2dsmooth', 'arm_2dsmooth', 'swing_2dsmooth', 'bucket_2dsmooth',
                                       'boom_last_ae', 'arm_last_ae', 'swing_last_ae', 'bucket_last_ae',
                                       'boom_me', 'arm_me', 'swing_me', 'bucket_me',
                                       'mean_end_dis'])
    df = df.rename(index=lambda x: 'file_' + str(x + 1))
    df = df.rename(index={df.index[-1]: 'total'})
    df.to_csv(save_path2 + '_mae' + "{:.6f}".format(all_mae) + '_msre' + "{:.6f}".format(all_msre) + '_1d'
              + "{:.6f}".format(all_1dsmooth) + '_2d' + "{:.6f}".format(all_2dsmooth) +
              '_last' + "{:.6f}".format(all_last_mae) + '_mean' + "{:.6f}".format(all_me) +
              '_end' + "{:.6f}".format(total_mean_end) + '.csv')


if __name__ == '__main__':
    pid_plot()
    pid_analysis()

