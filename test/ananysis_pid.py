import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import shutil


folder_path = '../origin_noise_data/data_with_noise_test'
folder_path = os.path.abspath(folder_path)

save_fp = '../simulation/figure_pid_test_grf'
shutil.rmtree(save_fp, ignore_errors=True)
os.makedirs(save_fp, exist_ok=True)

save_path2 = '../simulation/outcome_data/pid_test_with'


def pid_plot():
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            pos_series = np.array(data['js'])
            target_series = np.array(data['target'])

            t_pos = range(len(data['time']))
            t_target = range(1, len(data['time']))

            fig, axs = plt.subplots(2, 4, figsize=(24, 8))

            axs[0, 0].plot(t_pos, pos_series[:, 0], label='pos_boom')
            axs[0, 0].plot(t_target, target_series[:-1, 0], label='target_boom')
            axs[0, 0].set_xlabel('step')
            axs[0, 0].set_ylabel('pos angle (rad)')
            axs[0, 0].legend()

            axs[0, 1].plot(t_pos, pos_series[:, 1], label='pos_arm')
            axs[0, 1].plot(t_target, target_series[:-1, 1], label='target_arm')
            axs[0, 1].set_xlabel('step')
            axs[0, 1].set_ylabel('pos angle (rad)')
            axs[0, 1].legend()

            axs[0, 2].plot(t_pos, pos_series[:, 2], label='pos_swing')
            axs[0, 2].plot(t_target, target_series[:-1, 2], label='target_swing')
            axs[0, 2].set_xlabel('step')
            axs[0, 2].set_ylabel('pos angle (rad)')
            axs[0, 2].legend()

            axs[0, 3].plot(t_pos, pos_series[:, 3], label='pos_bucket')
            axs[0, 3].plot(t_target, target_series[:-1, 3], label='target_bucket')
            axs[0, 3].set_xlabel('step')
            axs[0, 3].set_ylabel('pos angle (rad)')
            axs[0, 3].legend()

            axs[1, 0].plot(t_target, pos_series[1:, 0] - target_series[:-1, 0], label='boom_error')
            axs[1, 0].set_xlabel('step')
            axs[1, 0].set_ylabel('error angle (rad)')
            axs[1, 0].legend()

            axs[1, 1].plot(t_target, pos_series[1:, 1] - target_series[:-1, 1], label='arm_error')
            axs[1, 1].set_xlabel('step')
            axs[1, 1].set_ylabel('error angle (rad)')
            axs[1, 1].legend()

            axs[1, 2].plot(t_target, pos_series[1:, 2] - target_series[:-1, 2], label='swing_error')
            axs[1, 2].set_xlabel('step')
            axs[1, 2].set_ylabel('error angle (rad)')
            axs[1, 2].legend()

            axs[1, 3].plot(t_target, pos_series[1:, 3] - target_series[:-1, 3], label='bucket_error')
            axs[1, 3].set_xlabel('step')
            axs[1, 3].set_ylabel('error angle (rad)')
            axs[1, 3].legend()

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
            target_series = np.array(data['target'])

            pos_complex = np.exp(1j * pos_series[1:, :])
            target_complex = np.exp(1j * target_series[:-1, :])
            abs_error = np.abs(np.angle(pos_complex / target_complex))

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
    pid_plot()
    # pid_analysis()

