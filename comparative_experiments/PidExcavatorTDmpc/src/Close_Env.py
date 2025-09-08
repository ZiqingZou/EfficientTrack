# from gym import spaces
import numpy as np
import time
from simulation.env import EnvCore_LocalPython
from simulation.pid_controller import PidController
import pandas as pd
import os
import shutil


class RLEnv:
    def __init__(self):
        # self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(4,), dtype=np.float64)  # 动作量4维
        # self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(332,), dtype=np.float64)  # 状态量332维

        self.ctl_env = EnvCore_LocalPython()
        self.controller = PidController()

        self.random = str(np.random.randint(1, 41))
        self.ref = np.load(os.path.dirname(__file__) + '/ref_traj/zc_interp_npy_train/zc_interp_train' + self.random + '.npy')
        self.idx = 0

        self.horizon = 20

        self.state = None
        self.previous_state = None
        self.target = None
        self.previous_target = None
        self.future_real_target = None

        self.data_save = {"target": [], "real_target": [], "js": [], "jv": [], "time": [], "action": []}

    def get_state(self, target_pos):
        pos_dict = self.ctl_env.get_dic_pos()
        now_pos = np.array(list(pos_dict.values()))
        vel_dict = self.ctl_env.get_dic_vel()
        now_vel = np.array(list(vel_dict.values()))

        state = np.concatenate((now_pos, now_vel))
        target = target_pos
        if self.idx == 0:
            previous_state = np.tile(state, self.horizon)
            previous_target = np.tile(target, self.horizon)
            future_real_target = self.ref[1:self.horizon + 1, :]
            future_real_target = future_real_target.flatten()
        else:
            previous_state = np.append(self.previous_state[8:], self.state)
            previous_target = np.append(self.previous_target[4:], self.target)
            if self.idx + self.horizon < self.ref.shape[0]:
                add_tg = self.ref[self.idx + self.horizon, :]
            else:
                add_tg = self.ref[-1, :]
            future_real_target = np.append(self.future_real_target[4:], add_tg)

        self.state = state
        self.previous_state = previous_state
        self.target = target
        self.previous_target = previous_target
        self.future_real_target = future_real_target

        rl_state = np.concatenate((previous_state, state, previous_target, target, future_real_target))
        return rl_state

    def reset(self, ref_traj=0, options='train'):
        if ref_traj == 0:
            self.random = str(np.random.randint(1, 41))
        else:
            self.random = str(ref_traj)
        self.ref = np.load(os.path.dirname(__file__) + '/ref_traj/zc_interp_npy_' + options + '/zc_interp_' + options + self.random + '.npy')
        self.idx = 0
        target_pos = self.ref[0, :]
        self.ctl_env.reset(target_pos.tolist() + [0, 0, 0, 0])

        self.data_save = {"target": [], "real_target": [], "js": [], "jv": [], "time": [], "action": []}
        print("\nreset!")
        print(options + " ref" + self.random)

        return self.get_state(target_pos)

    def get_reward(self):
        # PPO
        reward = 5.0 * (0.1 - np.sqrt(np.mean(np.square(self.state[:4] - self.ref[self.idx, :]))))  # MSRE
        # SAC
        # reward = 100 * (0.2 - np.sqrt(np.mean(np.square(self.state[:4] - self.ref[self.idx, :]))))  # MSRE
        return reward

    def step(self, a):
        a = a * 0.5

        pos_dict = self.ctl_env.get_dic_pos()
        now_pos = np.array(list(pos_dict.values()))
        vel_dict = self.ctl_env.get_dic_vel()
        now_vel = np.array(list(vel_dict.values()))

        real_target = self.ref[self.idx + 1, :]
        target_pos = real_target + a
        action = self.controller.pid_control(target_pos, now_pos)
        self.ctl_env.step(action.tolist())
        # print("step" + str(self.idx))
        self.idx += 1

        s_new = self.get_state(target_pos)

        reward = self.get_reward()

        done = bool(self.idx >= (self.ref.shape[0] - 1))
        # done = bool(self.idx >= 365)

        self.data_save["target"].append(target_pos)
        self.data_save["real_target"].append(real_target)
        self.data_save["js"].append(now_pos)
        self.data_save["jv"].append(now_vel)
        self.data_save["time"].append(time.time())
        self.data_save["action"].append(action)

        return s_new, reward, done, {}

    def save(self, iterr):
        file_name = 'zc_interp_train' + self.random
        os.makedirs(os.path.dirname(__file__) + '/save_data', exist_ok=True)
        np.save(os.path.dirname(__file__) + '/save_data/{}_{}.npy'.format(iterr, file_name), self.data_save)
        df = pd.DataFrame(self.data_save)
        csv_file = os.path.dirname(__file__) + '/save_data/{}_{}.csv'.format(iterr, file_name)
        df.to_csv(csv_file, index=False)
        print('\nsaved', time.localtime())

    def save_test(self, datanum):
        # timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        file_name = 'zc_interp_test' + self.random
        if self.random == '1':
            shutil.rmtree(os.path.dirname(__file__) + '/test_data_test/' + str(datanum), ignore_errors=True)
            os.makedirs(os.path.dirname(__file__) + '/test_data_test/' + str(datanum), exist_ok=True)
        np.save(os.path.dirname(__file__) + '/test_data_test/' + str(datanum) + '/{}.npy'.format(file_name), self.data_save)
        df = pd.DataFrame(self.data_save)
        csv_file = os.path.dirname(__file__) + '/test_data_test/' + str(datanum) + '/{}.csv'.format(file_name)
        df.to_csv(csv_file, index=False)
        print('\nsaved', time.localtime())


