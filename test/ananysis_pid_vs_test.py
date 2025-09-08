import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import shutil
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch


def find_matching_files(folder1, folder2, match_len):
    files1 = glob.glob(os.path.join(folder1, '*.npy'))
    files2 = glob.glob(os.path.join(folder2, '*.npy'))

    prefix_dict = {}

    for file in files1:
        filename = os.path.basename(file)
        prefix = filename[:match_len]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = [file]
        else:
            prefix_dict[prefix].append(file)

    matching_files = []
    for file in files2:
        filename = os.path.basename(file)
        prefix = filename[:match_len]
        if prefix in prefix_dict:
            matching_files.append((prefix_dict[prefix], file))

    return matching_files


def pid_plot():
    folder1 = '../data/real_excavator_data/data_without_noise_test'
    folder1 = os.path.abspath(folder1)
    folder2 = 'test_data_test'
    folder2 = os.path.abspath(folder2)

    save_fp = '../figure/real_plot'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)

    match_len = 17
    matching_files = find_matching_files(folder1, folder2, match_len)

    for files in matching_files:
        for file1 in files[0]:
            file2 = files[1]

            print(f"Matching files: {file1} and {file2}")

            if os.path.basename(file1)[:match_len] == 'zc_interp_test4_2':

                data_pid = np.load(file1, allow_pickle=True).item()
                data_policy = np.load(file2, allow_pickle=True).item()

                pid_pos_series = np.array(data_pid['js']) * 180 / np.pi
                policy_pos_series = np.array(data_policy['js']) * 180 / np.pi
                target_series = np.array(data_pid['real_target']) * 180 / np.pi
                policy_series = np.array(data_policy['target']) * 180 / np.pi

                t_pos = [ele * 0.05 for ele in list(range(len(data_pid['time'])))]
                t_target = [ele * 0.05 for ele in list(range(1, len(data_pid['time'])))]

                fig, axs = plt.subplots(1, 4, figsize=(24, 4))

                axs[0].plot(t_target, target_series[:-1, 0], label='reference', color='#85A79C', linestyle='--', linewidth=3)
                axs[0].plot(t_pos, pid_pos_series[:, 0], label='PD controller', color='#478058', linewidth=2)
                axs[0].plot(t_target, policy_series[:-1, 0], label='adjusted ref.', color='#808080', linewidth=1)
                axs[0].plot(t_pos, policy_pos_series[:, 0], label='ours', color='#CC5F5A', linewidth=2)
                axs[0].set_xlabel('time (s)', fontsize=18)
                axs[0].set_ylabel('boom angle (°)', fontsize=18)
                axs[0].tick_params(axis='both', which='major', labelsize=12)
                axs[0].legend(fontsize=15)

                axs[1].plot(t_target, target_series[:-1, 1], label='reference', color='#85A79C', linestyle='--', linewidth=3)
                axs[1].plot(t_pos, pid_pos_series[:, 1], label='PD controller', color='#478058', linewidth=2)
                axs[1].plot(t_target, policy_series[:-1, 1], label='adjusted ref.', color='#808080', linewidth=1)
                axs[1].plot(t_pos, policy_pos_series[:, 1], label='ours', color='#CC5F5A', linewidth=2)
                axs[1].set_xlabel('time (s)', fontsize=18)
                axs[1].set_ylabel('arm angle (°)', fontsize=18)
                axs[1].tick_params(axis='both', which='major', labelsize=12)
                axs[1].legend(fontsize=15)

                axs[2].plot(t_target, target_series[:-1, 2], label='reference', color='#85A79C', linestyle='--', linewidth=3)
                axs[2].plot(t_pos, pid_pos_series[:, 2], label='PD controller', color='#478058', linewidth=2)
                axs[2].plot(t_target, policy_series[:-1, 2], label='adjusted ref.', color='#808080', linewidth=1)
                axs[2].plot(t_pos, policy_pos_series[:, 2], label='ours', color='#CC5F5A', linewidth=2)
                axs[2].set_xlabel('time (s)', fontsize=18)
                axs[2].set_ylabel('bucket angle (°)', fontsize=18)
                axs[2].tick_params(axis='both', which='major', labelsize=12)
                axs[2].legend(fontsize=15)

                axs[3].plot(t_target, target_series[:-1, 3], label='reference', color='#85A79C', linestyle='--', linewidth=3)
                axs[3].plot(t_pos, pid_pos_series[:, 3], label='PD controller', color='#478058', linewidth=2)
                axs[3].plot(t_target, policy_series[:-1, 3], label='adjusted ref.', color='#808080', linewidth=1)
                axs[3].plot(t_pos, policy_pos_series[:, 3], label='ours', color='#CC5F5A', linewidth=2)
                axs[3].set_xlabel('time (s)', fontsize=18)
                axs[3].set_ylabel('swing angle (°)', fontsize=18)
                axs[3].tick_params(axis='both', which='major', labelsize=12)
                axs[3].legend(fontsize=15)

                plt.tight_layout()
                plt.savefig(save_fp + '/intrep_target_' + os.path.basename(file1)[:match_len] + '.png')
                plt.close(fig)


if __name__ == '__main__':
    pid_plot()
    