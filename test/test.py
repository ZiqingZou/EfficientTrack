from network_model.parser import parse_config
from simulation.collector import DataCollector
from simulation.pid_controller import PidController
from simulation.env import EnvCore_LocalPython
from network_model.controller import Controller
from network_model.predictor import Predictor
import shutil
import os


cfg = parse_config('../config.yaml')


def main():
    env = EnvCore_LocalPython()
    controller = PidController()
    policy = Controller(cfg)
    policy.load('../offline_train/state_dict/200000_controller.pth')
    model = Predictor(cfg)
    model.load('../offline_train/state_dict/200000_predictor.pth')

    name = 'zc_interp_train'
    save_fp = './test_data_train'
    shutil.rmtree(save_fp)
    os.makedirs(save_fp, exist_ok=True)
    noise = 0.0
    for i in range(1, 41):
        collector = DataCollector(env, controller, name + str(i), save_fp, noise, "train")
        collector.loopy(policy, model)
        print(name + str(i) + " done!\n")

    name = 'zc_interp_test'
    save_fp = './test_data_test'
    shutil.rmtree(save_fp, ignore_errors=True)
    os.makedirs(save_fp, exist_ok=True)
    noise = 0.0
    for i in range(1, 9):
        collector = DataCollector(env, controller, name + str(i), save_fp, noise, "test")
        collector.loopy(policy=policy, model=model)
        print(name + str(i) + " done!\n")


if __name__ == '__main__':
    main()
