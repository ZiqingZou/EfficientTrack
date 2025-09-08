from network_model.parser import parse_config
from network_model.predictor import Predictor
from network_model.controller import Controller
from online_train.buffer import Buffer

from Open_Env import RLEnv

cfg = parse_config('../config.yaml')


if __name__ == "__main__":
    env = RLEnv()

    predictor = Predictor(cfg)
    predictor.load('../online_train/state_dict/18000_predictor.pth')
    controller = Controller(cfg)
    controller.load('../online_train/state_dict/18000_controller.pth')

    for i in range(1, 9):
        s = env.reset(ref_traj=i, options='test')
        done = False

        while not done:
            a = controller.online_control(s).detach().cpu().numpy()
            s_, r, done, _ = env.step(a)
            s = s_

        env.save_test()
