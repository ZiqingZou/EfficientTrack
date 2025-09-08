import numpy as np
import os
import math
import joblib
import copy


def valve_func(x, a, b, c, d):
    return b / (1 + np.exp(-c*x+a)) + d


def pwm_velocity(pwm, position, valve_model, j_model):
    return abs(valve_func(abs(pwm) / 100, valve_model[0], valve_model[1], valve_model[2], valve_model[3]) * np.polyval(j_model, position))


class EnvCore_LocalPython:
    def __init__(self):
        super().__init__()
        self.parallel_num = 1
        self.core_type = 'python'
        self.joint_position = np.array([[0.0] * 4])
        self.joint_velocity = np.array([[0.0] * 4])
        self.step_num = 0
        self.episode_id = 0
        self.time_interval_mean = np.array([[0.05]])
        self.time_interval = None
        self.pos_limit = np.array([[-0.4, -1.0, -1.0, -math.pi], [0.8, 1.1, 3, math.pi]])
        print(os.path.dirname(__file__))
        self.vel_cal = [[
                joblib.load(os.path.dirname(__file__) +
                            '/python_dynamic_files/{joint}_{n_p}_model.pkl'.format(joint=_, n_p=__))
                for __ in ['positive', 'negative']]
            for _ in ['boom', 'arm', 'swing']]
        self.valve_model = [
            [
                [11.858506809692232, 0.512567730315598, 21.86915957497452, -0.0007917425932205652],
                [8.931731925229315, 0.36658149083455016, 15.700213825626527, -0.002794938954199982]
            ],
			[
                [14.023240203780164, 0.8183660786758581, 25.106667075640413, 0.0016937385965325377],
                [13.301471156931091, 0.6985445420696443, 21.749372501996476, 0.002202294140167611]
            ],
            [
                [14.952651261917973, 1.2749258196268678, 28.417406629267962, -0.0010930530561161407],
                [13.581409898795217, 1.0442331432592302, 25.098200285462813, -0.0013277636870057657]
            ]
            ]
        self.j_model = [
            [
                [0.22455292, 0.1251744, -0.56642958, 0.62115276, -0.43994377, 0.79004756],
                [0.37712406, 0.79595915, -1.23805643, 0.88716494, -0.62352762, 1.07171297]
            ],
            [
                [0.14598575, 0.67122928, 0.04021848, 0.33858849, 0.09810778, 0.91324457],
                [-0.36816399,  0.499325,   -0.04911397,  0.43650892,  0.00799341,  1.07954839]
            ],
            [
                [0.1282508, 0.10203362, -0.36151386, 0.50794024, 0.09358899, 0.98092446],
                [-0.08686432, 0.31000578, -0.21916701, 0.40319928, -0.00116702, 0.99735403]
            ]
            ]

    def reset(self, pos_vel_seq=None, action_time_seq=None):
        assert len(pos_vel_seq) == 8
        reset_pos_vel = pos_vel_seq
        self.joint_position = np.array([reset_pos_vel[:4]]).astype(np.float64)
        self.joint_velocity = np.array([reset_pos_vel[4:]]).astype(np.float64)
        self.step_num = 0
        self.episode_id += 1
        print("Env Reset!")

    def step(self, action_list, time_interval=0.05):
        if type(time_interval) is float:
            time_interval = np.array([time_interval] * self.parallel_num).reshape(-1, 1)
        action = action_list
        self.compute_vel(action)
        self.compute_pos(time_interval[0][0])
        self.step_num += 1
        self.time_interval = np.array(time_interval)

        # pos_dict = self.get_dic_pos()
        # vel_dict = self.get_dic_vel()
        #
        # return pos_dict, vel_dict

    def compute_pos(self, time_interval):
        self.joint_position[0] += time_interval * self.joint_velocity[0]
        boom_arm_bucket, swing = self.joint_position[0][:3], self.joint_position[0][3]
        boom_arm_bucket = np.clip(boom_arm_bucket, self.pos_limit[0][:3], self.pos_limit[1][:3])
        swing = (swing + math.pi) % (math.pi * 2) - math.pi
        self.joint_position[0] = np.concatenate([boom_arm_bucket, [swing]])

    def compute_vel(self, pwm_ori):
        pwm = copy.deepcopy(pwm_ori)
        joint_vel = np.array([0.0] * 4)
        for id, value in enumerate(pwm[:3]):
            if value > 0:
                if self.joint_position[0][id] <= self.pos_limit[0][id]:
                    joint_vel[id] = 0.0
                else:
                    joint_vel[id] = -pwm_velocity(value, self.joint_position[0][id], self.valve_model[id][0], self.j_model[id][0])
            elif value < 0:
                if self.joint_position[0][id] >= self.pos_limit[1][id]:
                    joint_vel[id] = 0.0
                else:
                    joint_vel[id] = pwm_velocity(value, self.joint_position[0][id], self.valve_model[id][1], self.j_model[id][1])

        if pwm[3] > 0:
            pwm[3] = round(pwm[3])
            if pwm[3] <= 51:
                joint_vel[3] = 0.0
            elif pwm[3] >= 77:
                joint_vel[3] = 1.0
            else:
                x_in = np.array([self.joint_position[0][3], abs(pwm[3]) / 100]).reshape(1, -1)
                joint_vel[3] = self.vel_cal[2][0].predict(x_in).tolist()[0][0]
        elif pwm[3] < 0:
            pwm[3] = round(pwm[3])
            if abs(pwm[3]) <= 66:
                joint_vel[3] = 0.0
            elif abs(pwm[3]) >= 90:
                joint_vel[3] = -1.0
            else:
                x_in = np.array([self.joint_position[0][3], abs(pwm[3]) / 100]).reshape(1, -1)
                joint_vel[3] = self.vel_cal[2][1].predict(x_in).tolist()[0][0]
        self.joint_velocity[0] = joint_vel

    def get_dic_pos(self):
        return {
            "boom": self.joint_position[0][0],
            "arm": self.joint_position[0][1],
            "bucket": self.joint_position[0][2],
            "swing": self.joint_position[0][3]
        }

    def get_dic_vel(self):
        return {
            "boom": self.joint_velocity[0][0],
            "arm": self.joint_velocity[0][1],
            "bucket": self.joint_velocity[0][2],
            "swing": self.joint_velocity[0][3]
        }


if __name__ == '__main__':
    env = EnvCore_LocalPython()
    env.reset([0, 0, 0, 0, 0, 0, 0, 0])
    print(env.step([1000, 1000, 1000, 1000]))
    print(env.step([500, -500, 500, -600]))
    print(env.step([0, 10, -300, 0]))
