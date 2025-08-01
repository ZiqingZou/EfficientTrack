from network_model.parser import parse_config
from collector import DataCollector
from pid_controller import PidController
from env import EnvCore_LocalPython
from network_model.controller import Controller
from network_model.predictor import Predictor
import shutil
import os


cfg = parse_config('../config.yaml')


def main():
    env = EnvCore_LocalPython()
    controller = PidController()
    policy = Controller(cfg)
    policy.load('../online_train/state_dict/200000_controller.pth')
    model = Predictor(cfg)
    model.load('../online_train/state_dict/200000_predictor.pth')

    # name = 'zc_interp_train'
    # save_fp = './test_data_train'
    # shutil.rmtree(save_fp)
    # os.makedirs(save_fp, exist_ok=True)
    # noise = 0.0
    # for i in range(1, 41):
    #     collector = DataCollector(env, controller, name + str(i), save_fp, noise)
    #     collector.loopy(policy, model)
    #     print(name + str(i) + " done!\n")

    name = 'zc_interp_test'
    save_fp = '../online_train/test_data_test'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)
    noise = 0.0
    for i in range(1, 9):
        collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'test')
        collector.loopy(policy, model)
        print(name + str(i) + " done!\n")

    # name = 'zc_interp_train'
    # save_fp = './data_without_noise_train'
    # shutil.rmtree(save_fp, ignore_errors=True)
    # os.makedirs(save_fp, exist_ok=True)
    # noise = 0.0
    # for i in range(1, 41):
    #     collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'train')
    #     collector.loopy(policy=policy, model=model)
    #     print(name + str(i) + " done!\n")

    # save_fp = './data_with_noise_train'
    # shutil.rmtree(save_fp, ignore_errors=True)
    # os.makedirs(save_fp, exist_ok=True)
    # for x in range(1):
    #     time.sleep(1)
    #     noise = (x ** 2) * 0.0005
    #     # noise = 0.3
    #     # noise = x * 0.01
    #     for i in range(1, 41):
    #         collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'train')
    #         collector.loopy(policy=policy, model=model)
    #         print(name + str(i) + " done!\n")

    name = 'zc_interp_test'
    save_fp = './data_without_noise_test'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)
    noise = 0.0
    for i in range(1, 9):
        collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'test')
        collector.loopy(policy=policy, model=model)
        print(name + str(i) + " done!\n")

    # save_fp = './data_with_noise_test'
    # shutil.rmtree(save_fp, ignore_errors=True)
    # os.makedirs(save_fp, exist_ok=True)
    # for x in range(1):
    #     time.sleep(1)
    #     noise = (x ** 2) * 0.0005
    #     # noise = 0.3
    #     # noise = x * 0.01
    #     for i in range(1, 9):
    #         collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'test')
    #         collector.loopy(policy=policy, model=model)
    #         print(name + str(i) + " done!\n")


if __name__ == '__main__':
    main()
