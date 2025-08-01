import wandb

from network_model.parser import parse_config
from network_model.predictor import Predictor
from network_model.controller import Controller
from online_train.buffer import Buffer

from Close_Env import RLEnv

cfg = parse_config('../config.yaml')
wandb.init(project='online', name='CloseEnv-SuperTrack-in1000')


def train_predictor(buffer, predictor, base_predictor):
    if predictor.iteration == 0:
        print('\n\nTraining predictor...')
        print('Total params number:', predictor.total_params, '\n')

    _, metrics, tsm = predictor.update(buffer, base_predictor)
    wandb.log({'predictor': metrics})

    if predictor.iteration % 1000 == 0:
        wandb.log({'pos_mae_iter' + str(predictor.iteration):
                   wandb.plot.line_series(
                       xs=list(range(1, predictor.horizon + 1)),
                       ys=[tsm["pos_1"], tsm["pos_2"], tsm["pos_3"], tsm["pos_4"], tsm["pos_avg"]],
                       keys=["pos_1", "pos_2", "pos_3", "pos_4", "pos_avg"],
                       xname='time_step',
                       title='predictor pos_mae_iter' + str(predictor.iteration))})
        wandb.log({'vel_mae_iter' + str(predictor.iteration):
                   wandb.plot.line_series(
                       xs=list(range(1, predictor.horizon + 1)),
                       ys=[tsm["vel_1"], tsm["vel_2"], tsm["vel_3"], tsm["vel_4"], tsm["vel_avg"]],
                       keys=["vel_1", "vel_2", "vel_3", "vel_4", "vel_avg"],
                       xname='time_step',
                       title='predictor vel_mae_iter' + str(predictor.iteration))})
        predictor.save('./state_dict/' + str(predictor.iteration) + '_predictor.pth')
        print('\nTraining predictor done!')
        print('Total iterations:', predictor.iteration, '\n\n')
    return


def train_controller(buffer, predictor, controller, base_controller):
    if controller.iteration == 0:
        print('\n\nTraining controller...')
        print('Total params number:', controller.total_params, '\n')

    _, metrics, tsm = controller.update(buffer, predictor, base_controller)
    wandb.log({'controller': metrics})

    if controller.iteration % 1000 == 0:
        wandb.log({'delta_target_iter' + str(controller.iteration):
                  wandb.plot.line_series(
                      xs=list(range(1, controller.horizon + 1)),
                      ys=[tsm["dt_1"], tsm["dt_2"], tsm["dt_3"], tsm["dt_4"], tsm["dt_avg"]],
                      keys=["delta_target_1", "delta_target_2",
                            "delta_target_3", "delta_target_4", "delta_target_avg"],
                      xname='time_step',
                      title='controller delta_target_iter' + str(controller.iteration))})
        wandb.log({'tracking_mae_iter' + str(controller.iteration):
                  wandb.plot.line_series(
                      xs=list(range(1, controller.horizon + 1)),
                      ys=[tsm["mae_1"], tsm["mae_2"], tsm["mae_3"], tsm["mae_4"], tsm["mae_avg"]],
                      keys=["tracking_mae_1", "tracking_mae_2",
                            "tracking_mae_3", "tracking_mae_4", "tracking_mae_avg"],
                      xname='time_step',
                      title='controller tracking_mae_iter' + str(controller.iteration))})
        controller.save('./state_dict/' + str(controller.iteration) + '_controller.pth')

        print('\nTraining controller done!')
        print('Total iterations:', controller.iteration, '\n\n')
    return


if __name__ == "__main__":
    env = RLEnv()
    buffer = Buffer(buffer_size=cfg.buffer_size, horizon=cfg.horizon,
                    state_dim=cfg.state_dim, target_dim=cfg.target_dim, batch_size=cfg.batch_size)

    predictor = Predictor(cfg)
    # predictor.load('./state_dict_ex4-mix11/190000_predictor.pth')
    base_predictor = None
    controller = Controller(cfg)
    # controller.load('./state_dict_ex4-mix11/190000_controller.pth')
    base_controller = None

    controller.save('./state_dict/0_controller.pth')
    predictor.save('./state_dict/0_predictor.pth')

    for i in range(0, 100000):
        # new episode
        s = env.reset()
        buffer.push(s)
        done = False

        while not done:
            a = controller.online_control(s).detach().cpu().numpy()
            s_, _, done, _ = env.step(a)
            buffer.push(s_)
            s = s_

        print('iter: ', i + 1)
        print('Total data number: %d' % buffer.num_push)
        print('iteration:', controller.iteration, '\n')

        print("*************TRAIN**************")

        if i < 10:
            for j in range(1):
                train_predictor(buffer, predictor, base_predictor)
                train_controller(buffer, predictor, controller, base_controller)
                wandb.log({"Training epoch": i})
        elif i < 50:
            for j in range(10):
                train_predictor(buffer, predictor, base_predictor)
                train_controller(buffer, predictor, controller, base_controller)
                wandb.log({"Training epoch": i})
        elif i < 100:  # 3000
            for j in range(30):
                train_predictor(buffer, predictor, base_predictor)
                train_controller(buffer, predictor, controller, base_controller)
                wandb.log({"Training epoch": i})
        else:
            for j in range(50):
                train_predictor(buffer, predictor, base_predictor)
                train_controller(buffer, predictor, controller, base_controller)
                wandb.log({"Training epoch": i})
        print("***********TRAIN OVER***********")

    wandb.finish()
