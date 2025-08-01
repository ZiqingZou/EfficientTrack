import time

from collector import DataCollector
from pid_controller import PidController
from env import EnvCore_LocalPython
import shutil
import os


def main():
    env = EnvCore_LocalPython()
    controller = PidController()

    name = 'zc_interp_train'
    save_fp = './data_without_noise_train'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)
    noise = 0.0
    for i in range(1, 41):
        collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'train')
        collector.loopy(policy=None, model=None)
        print(name + str(i) + " done!\n")

    save_fp = './data_with_noise_train'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)
    for x in range(10):
        time.sleep(1)
        # noise = (x ** 2) * 0.0005
        noise = 0.3
        # noise = x * 0.01
        for i in range(1, 41):
            collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'train')
            collector.loopy(policy=None, model=None)
            print(name + str(i) + " done!\n")

    name = 'zc_interp_test'
    save_fp = './data_without_noise_test'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)
    noise = 0.0
    for i in range(1, 9):
        collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'test')
        collector.loopy(policy=None, model=None)
        print(name + str(i) + " done!\n")

    save_fp = './data_with_noise_test'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)
    for x in range(10):
        time.sleep(1)
        # noise = (x ** 2) * 0.0005
        noise = 0.3
        # noise = x * 0.01
        for i in range(1, 9):
            collector = DataCollector(env, controller, name + str(i), save_fp, noise, 'test')
            collector.loopy(policy=None, model=None)
            print(name + str(i) + " done!\n")


if __name__ == '__main__':
    main()
