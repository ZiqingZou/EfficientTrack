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
    folder1 = '../data/real_excavator_data/data_without_noise_test'  # '../data/simulation_data/data_without_noise_test'  #
    folder1 = os.path.abspath(folder1)
    folder2 = '../real_excavator/data_outcome_test_round3'  # '../ex4-mix-2/test_data_test'  #
    folder2 = os.path.abspath(folder2)

    save_fp = '../figure/real_plot'  # '../ex4-mix-2/figure_pid_vs_test_test'  #
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
                # axins = axs[0].inset_axes((0.33, 0.1, 0.25, 0.28))
                # axins.plot(t_target, target_series[:-1, 0], label='reference', color='#85A79C', linestyle='--', linewidth=3)
                # axins.plot(t_pos, pid_pos_series[:, 0], label='PD controller', color='#478058', linewidth=2)
                # axins.plot(t_target, policy_series[:-1, 0], label='adjusted ref.', color='#D3D3D3', linewidth=1)
                # axins.plot(t_pos, policy_pos_series[:, 0], label='ours(Round3)', color='#CC5F5A', linewidth=2)
                # x0 = 13.3
                # x1 = 14.3
                # y0 = 6
                # y1 = 10
                # axins.set_xlim(x0, x1)
                # axins.set_ylim(y0, y1)
                # sx = [x0, x1, x1, x0, x0]
                # sy = [y0, y0, y1, y1, y0]
                # axs[0].plot(sx, sy, 'black', linewidth=0.5)
                # xy = (x0, y0)
                # xy2 = (x0, y1)
                # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", linewidth=0.5,
                #                       axesA=axins, axesB=axs[0])
                # axins.add_artist(con)
                # xy = (x1, y0)
                # xy2 = (x1, y1)
                # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", linewidth=0.5,
                #                       axesA=axins, axesB=axs[0])
                # axins.add_artist(con)

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

                # axins3 = axs[3].inset_axes((0.28, 0.20, 0.27, 0.28))
                #
                # axins3.plot(t_target, target_series[:-1, 3], label='reference', color='#85A79C', linestyle='--', linewidth=3)
                # axins3.plot(t_pos, pid_pos_series[:, 3], label='PD controller', color='#478058', linewidth=2)
                # axins3.plot(t_target, policy_series[:-1, 3], label='adjusted ref.', color='#D3D3D3', linewidth=1)
                # axins3.plot(t_pos, policy_pos_series[:, 3], label='ours(Round3)', color='#CC5F5A', linewidth=2)
                # x0 = 15.7
                # x1 = 16.5
                # y0 = -1.8
                # y1 = 6.9
                # axins3.set_xlim(x0, x1)
                # axins3.set_ylim(y0, y1)
                # sx = [x0, x1, x1, x0, x0]
                # sy = [y0, y0, y1, y1, y0]
                # axs[3].plot(sx, sy, 'black', linewidth=0.5)
                # xy = (x0, y0)
                # xy2 = (x1, y0)
                # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", linewidth=0.5,
                #                       axesA=axins3, axesB=axs[3])
                # axins3.add_artist(con)
                # xy = (x0, y1)
                # xy2 = (x1, y1)
                # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data", linewidth=0.5,
                #                       axesA=axins3, axesB=axs[3])
                # axins3.add_artist(con)

            # axs[1, 0].plot(t_target, pid_pos_series[1:, 0] - target_series[:-1, 0], label='pid_error', color='#478058')
            # axs[1, 0].plot(t_target, policy_pos_series[1:, 0] - target_series[:-1, 0], label='policy_error', color='#CC5F5A')
            # axs[1, 0].set_xlabel('step', fontsize=18)
            # axs[1, 0].set_ylabel('boom error (rad)', fontsize=18)
            # axs[1, 0].tick_params(axis='both', which='major', labelsize=12)
            # axs[1, 0].legend(fontsize=15)
            #
            # axs[1, 1].plot(t_target, pid_pos_series[1:, 1] - target_series[:-1, 1], label='pid_error', color='#478058')
            # axs[1, 1].plot(t_target, policy_pos_series[1:, 1] - target_series[:-1, 1], label='policy_error', color='#CC5F5A')
            # axs[1, 1].set_xlabel('step', fontsize=18)
            # axs[1, 1].set_ylabel('arm error (rad)', fontsize=18)
            # axs[1, 1].tick_params(axis='both', which='major', labelsize=12)
            # axs[1, 1].legend(fontsize=15)
            #
            # axs[1, 2].plot(t_target, pid_pos_series[1:, 2] - target_series[:-1, 2], label='pid_error', color='#478058')
            # axs[1, 2].plot(t_target, policy_pos_series[1:, 2] - target_series[:-1, 2], label='policy_error', color='#CC5F5A')
            # axs[1, 2].set_xlabel('step', fontsize=18)
            # axs[1, 2].set_ylabel('bucket error (rad)', fontsize=18)
            # axs[1, 2].tick_params(axis='both', which='major', labelsize=12)
            # axs[1, 2].legend(fontsize=15)
            #
            # axs[1, 3].plot(t_target, pid_pos_series[1:, 3] - target_series[:-1, 3], label='pid_error', color='#478058')
            # axs[1, 3].plot(t_target, policy_pos_series[1:, 3] - target_series[:-1, 3], label='policy_error', color='#CC5F5A')
            # axs[1, 3].set_xlabel('step', fontsize=18)
            # axs[1, 3].set_ylabel('swing error (rad)', fontsize=18)
            # axs[1, 3].tick_params(axis='both', which='major', labelsize=12)
            # axs[1, 3].legend(fontsize=15)

                plt.tight_layout()
                plt.savefig(save_fp + '/intrep_target_' + os.path.basename(file1)[:match_len] + '.png')
                plt.close(fig)



if __name__ == '__main__':
    pid_plot()