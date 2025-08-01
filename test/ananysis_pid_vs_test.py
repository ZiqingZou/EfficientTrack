import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import shutil

from sympy.printing.pretty.pretty_symbology import line_width


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
    folder1 = '../simulation/data_without_noise_train'
    folder1 = os.path.abspath(folder1)
    folder2 = '../simulation/test_data_train'
    folder2 = os.path.abspath(folder2)

    save_fp = '../simulation/figure_pid_vs_test_train'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)

    match_len = 17
    matching_files = find_matching_files(folder1, folder2, match_len)

    for files in matching_files:
        for file1 in files[0]:
            file2 = files[1]

            print(f"Matching files: {file1} and {file2}")

            data_pid = np.load(file1, allow_pickle=True).item()
            data_policy = np.load(file2, allow_pickle=True).item()

            pid_pos_series = np.array(data_pid['js'])
            policy_pos_series = np.array(data_policy['js'])
            target_series = np.array(data_pid['target'])

            t_pos = range(len(data_pid['time']))
            t_target = range(1, len(data_pid['time']))

            fig, axs = plt.subplots(2, 4, figsize=(24, 8))

            axs[0, 0].plot(t_target, target_series[:-1, 0], label='target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 0].plot(t_pos, pid_pos_series[:, 0], label='pid_pos', color='#478058', linewidth=2)
            axs[0, 0].plot(t_pos, policy_pos_series[:, 0], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 0].set_xlabel('step', fontsize=18)
            axs[0, 0].set_ylabel('boom angle (rad)', fontsize=18)
            axs[0, 0].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 0].legend(fontsize=15)

            axs[0, 1].plot(t_target, target_series[:-1, 1], label='target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 1].plot(t_pos, pid_pos_series[:, 1], label='pid_pos', color='#478058', linewidth=2)
            axs[0, 1].plot(t_pos, policy_pos_series[:, 1], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 1].set_xlabel('step', fontsize=18)
            axs[0, 1].set_ylabel('arm angle (rad)', fontsize=18)
            axs[0, 1].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 1].legend(fontsize=15)

            axs[0, 2].plot(t_target, target_series[:-1, 2], label='target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 2].plot(t_pos, pid_pos_series[:, 2], label='pid_pos', color='#478058', linewidth=2)
            axs[0, 2].plot(t_pos, policy_pos_series[:, 2], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 2].set_xlabel('step', fontsize=18)
            axs[0, 2].set_ylabel('swing angle (rad)', fontsize=18)
            axs[0, 2].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 2].legend(fontsize=15)

            axs[0, 3].plot(t_target, target_series[:-1, 3], label='target', color='#A2AFA6', linestyle='--', linewidth=3)
            axs[0, 3].plot(t_pos, pid_pos_series[:, 3], label='pid_pos', color='#478058', linewidth=2)
            axs[0, 3].plot(t_pos, policy_pos_series[:, 3], label='policy_pos', color='#CC5F5A', linewidth=2)
            axs[0, 3].set_xlabel('step', fontsize=18)
            axs[0, 3].set_ylabel('bucket angle (rad)', fontsize=18)
            axs[0, 3].tick_params(axis='both', which='major', labelsize=12)
            axs[0, 3].legend(fontsize=15)

            axs[1, 0].plot(t_target, pid_pos_series[1:, 0] - target_series[:-1, 0], label='pid_error', color='#478058')
            axs[1, 0].plot(t_target, policy_pos_series[1:, 0] - target_series[:-1, 0], label='policy_error', color='#CC5F5A')
            axs[1, 0].set_xlabel('step', fontsize=18)
            axs[1, 0].set_ylabel('boom error (rad)', fontsize=18)
            axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 0].legend(fontsize=15)

            axs[1, 1].plot(t_target, pid_pos_series[1:, 1] - target_series[:-1, 1], label='pid_error', color='#478058')
            axs[1, 1].plot(t_target, policy_pos_series[1:, 1] - target_series[:-1, 1], label='policy_error', color='#CC5F5A')
            axs[1, 1].set_xlabel('step', fontsize=18)
            axs[1, 1].set_ylabel('arm error (rad)', fontsize=18)
            axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 1].legend(fontsize=15)

            axs[1, 2].plot(t_target, pid_pos_series[1:, 2] - target_series[:-1, 2], label='pid_error', color='#478058')
            axs[1, 2].plot(t_target, policy_pos_series[1:, 2] - target_series[:-1, 2], label='policy_error', color='#CC5F5A')
            axs[1, 2].set_xlabel('step', fontsize=18)
            axs[1, 2].set_ylabel('swing error (rad)', fontsize=18)
            axs[1, 2].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 2].legend(fontsize=15)

            axs[1, 3].plot(t_target, pid_pos_series[1:, 3] - target_series[:-1, 3], label='pid_error', color='#478058')
            axs[1, 3].plot(t_target, policy_pos_series[1:, 3] - target_series[:-1, 3], label='policy_error', color='#CC5F5A')
            axs[1, 3].set_xlabel('step', fontsize=18)
            axs[1, 3].set_ylabel('bucket error (rad)', fontsize=18)
            axs[1, 3].tick_params(axis='both', which='major', labelsize=12)
            axs[1, 3].legend(fontsize=15)

            plt.tight_layout()
            plt.savefig(save_fp + '/intrep_target_' + os.path.basename(file1)[:match_len] + '.png')
            plt.close(fig)


if __name__ == '__main__':
    pid_plot()

