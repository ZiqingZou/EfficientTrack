import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil


folder_path = './test_data_test'
folder_path = os.path.abspath(folder_path)

save_fp = './figure_test_test'
shutil.rmtree(save_fp, ignore_errors=True)
os.makedirs(save_fp, exist_ok=True)

shutil.rmtree('./outcome_data', ignore_errors=True)
os.makedirs('./outcome_data', exist_ok=True)
save_path2 = './outcome_data/test_test'


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

            axs[0, 0].plot(t_target, target_series[:-1, 0], label='policy_target', color='#2878B5', linewidth=2)
            axs[0, 0].plot(t_target, real_target_series[:-1, 0], label='real_target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 0].plot(t_pos, pos_series[:, 0], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 0].set_xlabel('step', fontsize=18)
            axs[0, 0].set_ylabel('boom angle (rad)', fontsize=18)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 0].legend(fontsize=15)

            axs[0, 1].plot(t_target, target_series[:-1, 1], label='policy_target', color='#2878B5', linewidth=2)
            axs[0, 1].plot(t_target, real_target_series[:-1, 1], label='real_target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 1].plot(t_pos, pos_series[:, 1], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 1].set_xlabel('step', fontsize=18)
            axs[0, 1].set_ylabel('arm angle (rad)', fontsize=18)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 1].legend(fontsize=15)

            axs[0, 2].plot(t_target, target_series[:-1, 2], label='policy_target', color='#2878B5', linewidth=2)
            axs[0, 2].plot(t_target, real_target_series[:-1, 2], label='real_target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 2].plot(t_pos, pos_series[:, 2], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 2].set_xlabel('step', fontsize=18)
            axs[0, 2].set_ylabel('swing angle (rad)', fontsize=18)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 2].legend(fontsize=15)

            axs[0, 3].plot(t_target, target_series[:-1, 3], label='policy_target', color='#2878B5', linewidth=2)
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


def pid_analysis():

    file_mae = []
    file_maxe = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            pos_series = np.array(data['js'])
            real_target_series = np.array(data['real_target'])

            pos_complex = np.exp(1j * pos_series[1:, :])
            real_target_complex = np.exp(1j * real_target_series[:-1, :])
            abs_error = np.abs(np.angle(pos_complex / real_target_complex))

            mean_abs_error = np.mean(abs_error, axis=0)
            file_mae.append(mean_abs_error)
            max_abs_error = np.max(abs_error, axis=0)
            file_maxe.append(max_abs_error)
    file_mae = np.array(file_mae)
    file_maxe = np.array(file_maxe)

    total_mae = np.mean(file_mae, axis=0)
    total_maxe = np.mean(file_maxe, axis=0)

    all_mae = np.mean(total_mae)
    all_maxe = np.max(total_maxe)

    top = np.hstack((file_mae, file_maxe))
    bottom = np.hstack((total_mae, total_maxe))
    result = np.vstack((top, bottom))

    df = pd.DataFrame(result, columns=['boom_mae', 'arm_mae', 'swing_mae', 'bucket_mae',
                                       'boom_maxe', 'arm_maxe', 'swing_maxe', 'bucket_maxe'])
    df = df.rename(index=lambda x: 'file_' + str(x + 1))
    df = df.rename(index={df.index[-1]: 'total'})
    df.to_csv(save_path2 + '_mae' + str(all_mae) + '_maxe' + str(all_maxe) + '.csv')


if __name__ == '__main__':
    # pid_plot()
    pid_analysis()

