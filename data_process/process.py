import os
import random
import shutil
import numpy as np
import pandas as pd
import torch

from network_model.space_change import rad_to_sincos, rad_to_sincos_special_v


# 1: read pos sequence from for_test and convert to .npy ----------------------------
def create_for_test_npy():
    folder_path = 'data_process/for_test'
    folder_path = os.path.abspath(folder_path)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    save_path = 'data_collect/for_test_npy'
    save_path = os.path.abspath(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        data = df.to_numpy()

        data = data[:, 4:8]  # only read pos information

        npy_file = os.path.join(save_path, os.path.splitext(csv_file)[0] + '.npy')
        np.save(npy_file, data)


# 2: read for_test_npy and verify if all pos between -pi and pi ----------------------------
def verify_for_test_npy():
    folder_path = 'data_collect/for_test_npy'
    folder_path = os.path.abspath(folder_path)

    flag = False
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path)
            if np.any(data > np.pi) or np.any(data < -np.pi):
                print(f'{filename} contains elements greater than pi or less than -pi')
                flag = True

    print(flag)


# 3: convert data_with/out_noise time to 0 in the start ----------------------------
def convert_time_from_zero():
    folder_path = '../data_collect/data_with_noise'
    folder_path = os.path.abspath(folder_path)

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path, allow_pickle=True).item()
            time_series = data['time']
            t0 = time_series[0]
            new_time_series = [x - t0 for x in time_series]
            data['time'] = new_time_series
            np.save(file_path, data)
            df = pd.DataFrame(data)
            csv_file = os.path.splitext(file_path)[0] + '.csv'
            df.to_csv(csv_file, index=False)


# 4: convert all npy to torch.Tensor dataset ----------------------------
def convert_npy_to_torch_dataset():
    H = 20  # horizon
    freq = 20

    save_path = '../origin_noise_data/train_dataset/'  # simulation
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path)

    ty = ''  # '_open'
    if (ty == '_open') or (ty == '_mini_open'):
        idd = 'action'
    else:
        idd = 'target'

    for tp in ['with', 'without']:
        folder_path = '../origin_noise_data/data_' + tp + '_noise_train'  #data_outcome_test_round3_newtraj'  #
        folder_path = os.path.abspath(folder_path)

        # compute total length
        total_length = 0
        for filename in os.listdir(folder_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(folder_path, filename)
                data = np.load(file_path, allow_pickle=True).item()
                time_series = data['time']
                length = len(time_series)
                total_length += length
        print(total_length)

        state = torch.zeros(total_length, 2 * H + 1, 8, requires_grad=False)  # with:8257, without:8235, mix:16492
        target = torch.zeros(total_length, 3 * H, 4, requires_grad=False)
        real_target = torch.zeros(total_length, 3 * H, 4, requires_grad=False)

        total_index = 0
        for filename in os.listdir(folder_path):
            if filename.endswith('.npy'):
                file_path = os.path.join(folder_path, filename)
                data = np.load(file_path, allow_pickle=True).item()
                file_length = len(data['time'])

                file_index = 0
                for _ in range(file_length):
                    # state
                    state[total_index, H, :4] = torch.tensor(data['js'][file_index])
                    state[total_index, H, 4:] = torch.tensor(data['jv'][file_index])
                    for i in range(H):
                        if file_index - i - 1 >= 0:
                            state[total_index, H - i - 1, :4] = torch.tensor(data['js'][file_index - i - 1])
                            state[total_index, H - i - 1, 4:] = torch.tensor(data['jv'][file_index - i - 1])
                        else:
                            state[total_index, H - i - 1, :4] = torch.tensor(data['js'][0])
                            state[total_index, H - i - 1, 4:] = torch.tensor(data['jv'][0])
                        if file_index + i + 1 < file_length:
                            state[total_index, H + i + 1, :4] = torch.tensor(data['js'][file_index + i + 1])
                            state[total_index, H + i + 1, 4:] = torch.tensor(data['jv'][file_index + i + 1])
                        else:
                            state[total_index, H + i + 1, :4] = torch.tensor(data['js'][file_length - 1])
                            state[total_index, H + i + 1, 4:] = torch.tensor(data['jv'][file_length - 1])

                    # target
                    target[total_index, H + 1, :] = torch.tensor(data[idd][file_index])
                    for i in range(H + 1):
                        if file_index - i - 1 >= 0:
                            target[total_index, H - i, :] = torch.tensor(data[idd][file_index - i - 1])
                        else:
                            target[total_index, H - i, :] = torch.tensor(data[idd][0])
                    for i in range(2 * H - 2):
                        if file_index + i + 1 < file_length:
                            target[total_index, H + i + 2, :] = torch.tensor(data[idd][file_index + i + 1])
                        else:
                            target[total_index, H + i + 2, :] = torch.tensor(data[idd][file_length - 1])
                    if idd == 'action':
                        target = target * 0.001

                    # real target
                    real_target[total_index, H + 1, :] = torch.tensor(data['real_target'][file_index])
                    for i in range(H + 1):
                        if file_index - i - 1 >= 0:
                            real_target[total_index, H - i, :] = torch.tensor(data['real_target'][file_index - i - 1])
                        else:
                            real_target[total_index, H - i, :] = torch.tensor(data['real_target'][0])
                    for i in range(2 * H - 2):
                        if file_index + i + 1 < file_length:
                            real_target[total_index, H + i + 2, :] = torch.tensor(data['real_target'][file_index + i + 1])
                        else:
                            real_target[total_index, H + i + 2, :] = torch.tensor(data['real_target'][file_length - 1])

                    total_index += 1
                    file_index += 1

        # state_sincos = rad_to_sincos(state)
        # target_sincos = rad_to_sincos(target)
        #
        # state_sincos_special_v = rad_to_sincos_special_v(state, freq)
        # target_sincos_special_v = target_sincos

        torch.save(state, save_path + 'state_' + tp + '_rad' + ty + '.pt')
        # torch.save(state_sincos, save_path + 'state_' + tp + '_sin-cos.pt')
        # torch.save(state_sincos_special_v, save_path + 'state_' + tp + '_sin-cos-special-v.pt')
        torch.save(target, save_path + 'target_' + tp + '_rad' + ty + '.pt')
        # torch.save(target_sincos, save_path + 'target_' + tp + '_sin-cos.pt')
        # torch.save(target_sincos_special_v, save_path + 'target_' + tp + '_sin-cos-special-v.pt')
        torch.save(real_target, save_path + 'real_target_' + tp + '_rad' + ty + '.pt')


    # mix
    state_with_rad = torch.load(save_path + 'state_with_rad' + ty + '.pt')
    # state_with_sincos = torch.load(save_path + 'state_with_sin-cos.pt')
    # state_with_sincos_special_v = torch.load(save_path + 'state_with_sin-cos-special-v.pt')
    target_with_rad = torch.load(save_path + 'target_with_rad' + ty + '.pt')
    # target_with_sincos = torch.load(save_path + 'target_with_sin-cos.pt')
    # target_with_sincos_special_v = torch.load(save_path + 'target_with_sin-cos-special-v.pt')
    real_target_with_rad = torch.load(save_path + 'real_target_with_rad' + ty + '.pt')

    state_without_rad = torch.load(save_path + 'state_without_rad' + ty + '.pt')
    # state_without_sincos = torch.load(save_path + 'state_without_sin-cos.pt')
    # state_without_sincos_special_v = torch.load(save_path + 'state_without_sin-cos-special-v.pt')
    target_without_rad = torch.load(save_path + 'target_without_rad' + ty + '.pt')
    # target_without_sincos = torch.load(save_path + 'target_without_sin-cos.pt')
    # target_without_sincos_special_v = torch.load(save_path + 'target_without_sin-cos-special-v.pt')
    real_target_without_rad = torch.load(save_path + 'real_target_without_rad' + ty + '.pt')

    state_mix_rad = torch.cat((state_with_rad, state_without_rad), dim=0)
    # state_mix_sincos = torch.cat((state_with_sincos, state_without_sincos), dim=0)
    # state_mix_sincos_special_v = torch.cat((state_with_sincos_special_v, state_without_sincos_special_v), dim=0)
    target_mix_rad = torch.cat((target_with_rad, target_without_rad), dim=0)
    # target_mix_sincos = torch.cat((target_with_sincos, target_without_sincos), dim=0)
    # target_mix_sincos_special_v = torch.cat((target_with_sincos_special_v, target_without_sincos_special_v), dim=0)
    real_target_mix_rad = torch.cat((real_target_with_rad, real_target_without_rad), dim=0)

    torch.save(state_mix_rad, save_path + 'state_mix_rad' + ty + '.pt')
    # torch.save(state_mix_sincos, save_path + 'state_mix_sin-cos.pt')
    # torch.save(state_mix_sincos_special_v, save_path + 'state_mix_sin-cos-special-v.pt')
    torch.save(target_mix_rad, save_path + 'target_mix_rad' + ty + '.pt')
    # torch.save(target_mix_sincos, save_path + 'target_mix_sin-cos.pt')
    # torch.save(target_mix_sincos_special_v, save_path + 'target_mix_sin-cos-special-v.pt')
    torch.save(real_target_mix_rad, save_path + 'real_target_mix_rad' + ty + '.pt')


# 5: split train set and test set ----------------------------
def train_test_split():
    for tp in ['with', 'without']:
        source_folder = '../data_collect/data_' + tp + '_noise'
        destination_folder_b = '../data_collect/data_' + tp + '_noise_test'
        destination_folder_c = '../data_collect/data_' + tp + '_noise_train'
        os.makedirs(destination_folder_b, exist_ok=True)
        os.makedirs(destination_folder_c, exist_ok=True)

        files = [f for f in os.listdir(source_folder) if f.endswith('.npy')]

        random_files = random.sample(files, 9)

        for file in random_files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder_b, file))

        remaining_files = set(files) - set(random_files)
        for file in remaining_files:
            shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder_c, file))


# 6: interpolate data to 20hz in zc_npy ----------------------------
def npy_interp():
    folder_path = './zc_npy'
    folder_path = os.path.abspath(folder_path)

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path)
            if np.any(data > np.pi) or np.any(data < -np.pi):
                print(f'{filename} contains elements greater than pi or less than -pi')

            data_len = len(data[:, 0])
            new_n = 2 * data_len - 1
            interpolated_array = np.zeros((new_n, 4))

            for col in range(4):
                x = np.arange(data_len)
                y = data[:, col]
                x_new = np.linspace(0, data_len - 1, new_n)
                interpolated_array[:, col] = np.interp(x_new, x, y)

            np.save('./zc_interp_npy/interp_' + filename, interpolated_array)
            print(f'{filename} interpolated.')


# 7: rename npy file to sequence ----------------------------
def file_rename():
    folder_path = './zc_interp_npy_test'
    folder_path = os.path.abspath(folder_path)

    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    files.sort()

    for i, filename in enumerate(files):
        new_name = f'zc_interp_test{i + 1}.npy'
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)

    print("Finished !")


# 8: clip dataset randomly to desire size
def clip_dataset():
    desire_size = 159048
    save_path = '../simulation/train_dataset/'

    # state_with_rad = torch.load(save_path + 'state_with_rad.pt')
    # target_with_rad = torch.load(save_path + 'target_with_rad.pt')
    # real_target_with_rad = torch.load(save_path + 'real_target_with_rad.pt')

    state_mix_rad = torch.load(save_path + 'state_mix_rad2.pt')
    target_mix_rad = torch.load(save_path + 'target_mix_rad2.pt')
    real_target_mix_rad = torch.load(save_path + 'real_target_mix_rad2.pt')

    idx = torch.randperm(state_mix_rad.size(0))[:desire_size]
    state_mix_radM = state_mix_rad[idx]
    target_mix_radM = target_mix_rad[idx]
    real_target_mix_radM = real_target_mix_rad[idx]

    torch.save(state_mix_radM, save_path + 'state_mix_radM.pt')
    torch.save(target_mix_radM, save_path + 'target_mix_radM.pt')
    torch.save(real_target_mix_radM, save_path + 'real_target_mix_radM.pt')

    desire_size = 17672

    idx = torch.randperm(state_mix_rad.size(0))[:desire_size]
    state_mix_radS = state_mix_rad[idx]
    target_mix_radS = target_mix_rad[idx]
    real_target_mix_radS = real_target_mix_rad[idx]

    torch.save(state_mix_radS, save_path + 'state_mix_radS.pt')
    torch.save(target_mix_radS, save_path + 'target_mix_radS.pt')
    torch.save(real_target_mix_radS, save_path + 'real_target_mix_radS.pt')

    # idx = torch.randperm(state_with_rad.size(0))[:desire_size]
    # state_with_radS = state_with_rad[idx]
    # target_with_radS = target_with_rad[idx]
    # real_target_with_radS = real_target_with_rad[idx]

    # torch.save(state_with_radS, save_path + 'state_with_radS.pt')
    # torch.save(target_with_radS, save_path + 'target_with_radS.pt')
    # torch.save(real_target_with_radS, save_path + 'real_target_with_radS.pt')


# 9: change rate of no-noise and noise data
def change_rate():
    no_noise_part = 2
    ty = '_open'
    save_path = '../simulation/train_dataset/'

    state_with_rad = torch.load(save_path + 'state_with_rad' + ty + '.pt')
    target_with_rad = torch.load(save_path + 'target_with_rad' + ty + '.pt')
    real_target_with_rad = torch.load(save_path + 'real_target_with_rad' + ty + '.pt')

    state_without_rad = torch.load(save_path + 'state_without_rad' + ty + '.pt')
    target_without_rad = torch.load(save_path + 'target_without_rad' + ty + '.pt')
    real_target_without_rad = torch.load(save_path + 'real_target_without_rad' + ty + '.pt')

    state_mix_rad = state_with_rad
    target_mix_rad = target_with_rad
    real_target_mix_rad = real_target_with_rad
    for _ in range(no_noise_part):
        state_mix_rad = torch.cat((state_without_rad, state_mix_rad), dim=0)
        target_mix_rad = torch.cat((target_without_rad, target_mix_rad), dim=0)
        real_target_mix_rad = torch.cat((real_target_without_rad, real_target_mix_rad), dim=0)

    torch.save(state_mix_rad, save_path + 'state_mix_rad' + ty + str(no_noise_part) + '.pt')
    torch.save(target_mix_rad, save_path + 'target_mix_rad' + ty + str(no_noise_part) + '.pt')
    torch.save(real_target_mix_rad, save_path + 'real_target_mix_rad' + ty + str(no_noise_part) + '.pt')


# 10: find max and min action ----------------------------
def find_max_and_min_action():
    for p2 in ['train', 'test']:
        for p1 in ['with', 'without']:
            folder_path = '../simulation/data_' + p1 + '_noise_' + p2
            folder_path = os.path.abspath(folder_path)
            max_action = -1
            min_action = 1
            for filename in os.listdir(folder_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(folder_path, filename)
                    data = np.load(file_path, allow_pickle=True).item()
                    action_series = data['action']
                    max_action = np.max([max_action, np.max(action_series)])
                    min_action = np.min([min_action, np.min(action_series)])
            print(p2, p1, max_action, min_action)


if __name__ == '__main__':
    convert_npy_to_torch_dataset()
    # find_max_and_min_action()
    # change_rate()

