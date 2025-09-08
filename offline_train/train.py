import wandb

from network_model.parser import parse_config
from network_model.predictor import Predictor
from network_model.controller import Controller
from offline_train.container import Container

import time

cfg = parse_config('../config.yaml')
wandb.init(project='EfficientTrack', name='simulation')

epoch_p, epoch_c = 1, 1


def train_predictor(container, predictor, base_predictor):
    print('\n\nTraining predictor...')
    print('Total params number:', predictor.total_params, '\n')

    for e in range(1, cfg.predictor_epochs + 1):
        tsm = None  # time_step_metrics

        epoch_end = False
        i = 0
        while not epoch_end:
            i += 1
            print('Epoch: {}, Iter: {}'.format(e, i))
            epoch_end, metrics, tsm = predictor.update(container, base_predictor)
            wandb.log({'predictor': metrics})

        if e % 50 == 0:
            wandb.log({'pos_mae_iter' + str(predictor.iteration):
                       wandb.plot.line_series(
                           xs=list(range(1, predictor.horizon + 1)),
                           ys=[tsm["pos_1"], tsm["pos_2"], tsm["pos_3"], tsm["pos_4"], tsm["pos_avg"]],
                           keys=["pos_1", "pos_2", "pos_3", "pos_4", "pos_avg"],
                           xname='time_step',
                           title='predictor pos_mae_epoch' + str(e))})
            wandb.log({'vel_mae_iter' + str(predictor.iteration):
                       wandb.plot.line_series(
                           xs=list(range(1, predictor.horizon + 1)),
                           ys=[tsm["vel_1"], tsm["vel_2"], tsm["vel_3"], tsm["vel_4"], tsm["vel_avg"]],
                           keys=["vel_1", "vel_2", "vel_3", "vel_4", "vel_avg"],
                           xname='time_step',
                           title='predictor vel_mae_epoch' + str(e))})
            predictor.save('./state_dict/' + str(predictor.iteration) + '_predictor_' + str(e) + '_' + str(i) + '.pth')

    print('\nTraining predictor done!')
    print('Total iterations:', predictor.iteration, '\n\n')
    return


def train_controller(container, predictor, controller, base_controller):
    print('\n\nTraining controller...')
    print('Total params number:', controller.total_params, '\n')

    for e in range(1, cfg.controller_epochs + 1):
        tsm = None  # time_step_metrics
        epoch_end = False
        i = 0
        while not epoch_end:
            i += 1
            print('Epoch: {}, Iter: {}'.format(e, i))
            epoch_end, metrics, tsm = controller.update(container, predictor, base_controller)
            wandb.log({'controller': metrics})

        if e % 50 == 0:
            wandb.log({'delta_target_iter' + str(controller.iteration):
                      wandb.plot.line_series(
                          xs=list(range(1, controller.horizon + 1)),
                          ys=[tsm["dt_1"], tsm["dt_2"], tsm["dt_3"], tsm["dt_4"], tsm["dt_avg"]],
                          keys=["delta_target_1", "delta_target_2",
                                "delta_target_3", "delta_target_4", "delta_target_avg"],
                          xname='time_step',
                          title='controller delta_target_epoch' + str(e))})
            wandb.log({'tracking_mae_iter' + str(controller.iteration):
                      wandb.plot.line_series(
                          xs=list(range(1, controller.horizon + 1)),
                          ys=[tsm["mae_1"], tsm["mae_2"], tsm["mae_3"], tsm["mae_4"], tsm["mae_avg"]],
                          keys=["tracking_mae_1", "tracking_mae_2",
                                "tracking_mae_3", "tracking_mae_4", "tracking_mae_avg"],
                          xname='time_step',
                          title='controller tracking_mae_epoch' + str(e))})
            controller.save('./state_dict/' + str(controller.iteration) + '_controller_' + str(e) + '_' + str(i) + '.pth')

    print('\nTraining controller done!')
    print('Total iterations:', controller.iteration, '\n\n')
    return


def train():
    state_fp = '../data/simulation_data/train_dataset/state_' + cfg.data_type + '_' + cfg.represent + '_newtraj.pt'  # simulation
    target_fp = '../data/simulation_data/train_dataset/target_' + cfg.data_type + '_' + cfg.represent + '_newtraj.pt'  # _mini
    real_target_fp = '../data/simulation_data/train_dataset/real_target_' + cfg.data_type + '_' + cfg.represent + '_newtraj.pt'
    container = Container(state_path=state_fp, target_path=target_fp,
                          real_target_path=real_target_fp, batch_size=cfg.batch_size)

    predictor = Predictor(cfg)
    # predictor.load('./state_dict/12000_predictor_4000_3.pth')  # './state_dict_real_round3/18000_predictor_2000_9.pth')
    # base_predictor = Predictor(cfg)
    base_predictor = None
    # base_predictor.load('./ex1-base_state_dict/190000_predictor_2000_95.pth')
    controller = Controller(cfg)
    # controller.load('./state_dict_real_round3/18000_controller_2000_9.pth')
    # base_controller = Controller(cfg)
    base_controller = None
    # base_controller.load('./ex1-base_state_dict/190000_controller_2000_95.pth')

    t0 = time.time()
    train_predictor(container, predictor, base_predictor)
    t1 = time.time()
    train_controller(container, predictor, controller, base_controller)
    t2 = time.time()
    wandb.log({"predictor_training_time": t1-t0, "controller_training_time": t2-t1})

    wandb.finish()
    return


def alternating_train_predictor(container, predictor, i):
    global epoch_p

    print('\nTraining predictor...')
    if i == 1:
        print('Total params number:', predictor.total_params, '\n')

    print('Epoch: {}, Total_Iter: {}'.format(epoch_p, i))
    epoch_end, metrics, tsm = predictor.update(container)
    wandb.log({'predictor': metrics})

    if epoch_end:
        epoch_p += 1
        if epoch_p % 50 == 0:
            wandb.log({'pos_mae_iter' + str(predictor.iteration):
                       wandb.plot.line_series(
                           xs=list(range(1, predictor.horizon + 1)),
                           ys=[tsm["pos_1"], tsm["pos_2"], tsm["pos_3"], tsm["pos_4"], tsm["pos_avg"]],
                           keys=["pos_1", "pos_2", "pos_3", "pos_4", "pos_avg"],
                           xname='time_step',
                           title='predictor pos_mae_epoch' + str(epoch_p))})
            wandb.log({'vel_mae_iter' + str(predictor.iteration):
                       wandb.plot.line_series(
                           xs=list(range(1, predictor.horizon + 1)),
                           ys=[tsm["vel_1"], tsm["vel_2"], tsm["vel_3"], tsm["vel_4"], tsm["vel_avg"]],
                           keys=["vel_1", "vel_2", "vel_3", "vel_4", "vel_avg"],
                           xname='time_step',
                           title='predictor vel_mae_epoch' + str(epoch_p))})

            predictor.save('./state_dict/' + str(predictor.iteration) +
                           '_predictor_' + str(epoch_p) + '_total' + str(i) + '.pth')

    print('Training predictor done!\n')
    return


def alternating_train_controller(container, predictor, controller, i):
    global epoch_c

    print('\nTraining controller...')
    if i == 1:
        print('Total params number:', controller.total_params, '\n')

    print('Epoch: {}, Total_Iter: {}'.format(epoch_c, i))
    epoch_end, metrics, tsm = controller.update(container, predictor)
    wandb.log({'controller': metrics})

    if epoch_end:
        epoch_c += 1
        if epoch_c % 50 == 0:
            wandb.log({'delta_target_iter' + str(controller.iteration):
                      wandb.plot.line_series(
                          xs=list(range(1, controller.horizon + 1)),
                          ys=[tsm["dt_1"], tsm["dt_2"], tsm["dt_3"], tsm["dt_4"], tsm["dt_avg"]],
                          keys=["delta_target_1", "delta_target_2",
                                "delta_target_3", "delta_target_4", "delta_target_avg"],
                          xname='time_step',
                          title='controller delta_target_epoch' + str(epoch_c))})
            wandb.log({'tracking_mae_iter' + str(controller.iteration):
                      wandb.plot.line_series(
                          xs=list(range(1, controller.horizon + 1)),
                          ys=[tsm["mae_1"], tsm["mae_2"], tsm["mae_3"], tsm["mae_4"], tsm["mae_avg"]],
                          keys=["tracking_mae_1", "tracking_mae_2",
                                "tracking_mae_3", "tracking_mae_4", "tracking_mae_avg"],
                          xname='time_step',
                          title='controller tracking_mae_epoch' + str(epoch_c))})

            controller.save('./state_dict/' + str(controller.iteration) +
                            '_controller_' + str(epoch_c) + '_total' + str(i) + '.pth')

    print('Training controller done!\n')
    return


def alternating_train():
    state_fp = '../simulation/train_dataset/state_' + cfg.data_type + '_' + cfg.represent + '.pt'
    target_fp = '../simulation/train_dataset/target_' + cfg.data_type + '_' + cfg.represent + '.pt'
    container_p = Container(state_fp, target_fp)
    container_c = Container(state_fp, target_fp)

    predictor = Predictor(cfg)
    controller = Controller(cfg)

    for iter_ in range(1, cfg.alternating_iters + 1):
        alternating_train_predictor(container_p, predictor, iter_)
        alternating_train_controller(container_c, predictor, controller, iter_)
        alternating_train_controller(container_c, predictor, controller, iter_)

    return


if __name__ == '__main__':
    train()
    # alternating_train()
    wandb.finish()
