import torch
import torch.nn as nn
import torch.nn.functional as F

from network_model.network import mlp, mlp_norm
from network_model.weight_init import weight_init
from network_model.space_change import sincos_to_rad, sincos_to_rad_special_v
from network_model.angle_error import angle_error
from offline_train.container import Container

from online_train.buffer import Buffer


class Predictor(nn.Module):
    """
    Predictor model architecture.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.device)
        self.horizon = cfg.horizon
        self.train_horizon = cfg.train_horizon
        self.discount = cfg.predictor_discount
        self.k_pos = cfg.predictor_k_pos
        self.represent = cfg.represent
        self.frequency = cfg.frequency
        self.max_range = cfg.predictor_max_range
        self.k_stable = cfg.predictor_stable_k

        self.iteration = 0

        predict_dim = (cfg.horizon + 1) * cfg.state_dim + (cfg.horizon + 2) * cfg.target_dim
        if cfg.predictor_type == 'mlp':
            self._predictor = mlp(predict_dim, cfg.predictor_hidden_depth * [cfg.predictor_hidden_dim],
                                  cfg.state_dim, tanh_out=True)
        elif cfg.predictor_type == 'mlp_norm':
            self._predictor = mlp_norm(predict_dim, cfg.predictor_hidden_depth * [cfg.predictor_hidden_dim],
                                       cfg.state_dim, cfg.predictor_dropout, tanh_out=True)
        else:
            raise ValueError('Unknown predictor type: {}'.format(cfg.predictor_type))
        self.apply(weight_init)
        self._predictor = self._predictor.to(self.device)

        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.predictor_lr, weight_decay=cfg.predictor_weight_decay)
        self.max_norm = cfg.predictor_max_norm

        self.eval()

    @property
    def total_params(self):
        """
        Total params number of params requires grad.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, state, previous_state, target, previous_target, next_target):
        """
        Predicts the next state given

        Args:
            current state(Tensor(batch_size, state_dim)),
            previous_state(Tensor(batch_size, horizon, state_dim)),
            current target(Tensor(batch_size, target_dim)),
            previous_target(Tensor(batch_size, horizon, target_dim)),
            and the next_target(Tensor(batch_size, target_dim)).
        
        Returns:
            next_state(Tensor(batch_size, state_dim)).
        """

        previous_state = previous_state.view(previous_state.shape[0], -1)
        previous_target = previous_target.view(previous_target.shape[0], -1)
        x = torch.cat([previous_state, state, previous_target, target, next_target], dim=-1)

        # # regularization
        # if self.represent == 'rad':
        #     torch.atan2(torch.sin(x), torch.cos(x))

        delta_state_predict = self.max_range * self._predictor(x)
        next_state_predict = state + delta_state_predict
        return next_state_predict, delta_state_predict
    
    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"predictor": self.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, weights_only=True)
        self.load_state_dict(state_dict["predictor"])

    def compute_loss(self, next_state_predict, next_state):
        """
        Compute loss in rad space between predicted and actual next state.

        Args:
            next_state_predict (torch.Tensor): Predicted next state (of shape (batch_size, state_dim)).
            next_state (torch.Tensor): Actual next state (of shape (batch_size, state_dim)).

        Returns:
            torch.Tensor: Mean squared error loss between predicted and actual next state.
        """
        if self.represent == 'rad':
            next_pos_predict, next_vel_predict = next_state_predict[:, :4], next_state_predict[:, 4:]
            next_pos, next_vel = next_state[:, :4], next_state[:, 4:]
        elif self.represent == 'sin-cos':
            next_pos_predict, next_vel_predict = sincos_to_rad(next_state_predict)
            next_pos, next_vel = sincos_to_rad(next_state)
        elif self.represent == 'sin-cos-special-v':
            next_pos_predict, next_vel_predict = sincos_to_rad_special_v(next_state_predict, self.frequency)
            next_pos, next_vel = sincos_to_rad_special_v(next_state, self.frequency)
        else:
            raise ValueError(f"Invalid represent: {self.represent}")

        pos_error = angle_error(next_pos_predict, next_pos)
        pos_loss = F.mse_loss(pos_error, torch.zeros_like(pos_error))  # dim and batch mean
        pos_1_error = angle_error(next_pos_predict[0], next_pos[0])
        pos_loss = pos_loss + 2 * F.mse_loss(pos_1_error, torch.zeros_like(pos_1_error))
        vel_loss = F.mse_loss(next_vel_predict, next_vel)  # dim and batch mean

        pos_mae = torch.mean(torch.abs(pos_error.detach()), dim=0)  # batch mean only
        vel_mae = torch.mean(torch.abs(next_vel_predict.detach() - next_vel), dim=0)  # batch mean only

        return pos_loss, vel_loss, pos_mae, vel_mae
    
    def sequence_update(self, previous, new):
        """
        Update previous_state and previous_target.

        Args:
            previous (Tensor(batch_size, horizon, state_dim/target_dim)),
            new (Tensor(batch_size, state_dim/target_dim)).

        Returns:
            updated (Tensor(batch_size, horizon, state_dim/target_dim)): previous_state/target for next step.
        """
        updated = torch.empty_like(previous)
        updated[:, :-1, :] = previous[:, 1:, :].clone()
        updated[:, -1, :] = new.clone()
        return updated

    def update(self, container: Container or Buffer, base_predictor=None):
        """
        Main update function. Corresponds to one iteration of model learning. 
        
        Args:
            container (data_process.container.Container): Data container.
            base_predictor (Predictor): Base model to keep near to.
        
        Returns:
            dict: Dictionary of training statistics.
        """
        state_sequence, target_sequence, _, epoch_end = container.sample()
        state_sequence = state_sequence.to(self.device).requires_grad_(False)
        target_sequence = target_sequence[:, :(2 * self.horizon + 1), :].to(self.device).requires_grad_(False)
        """
        None grad data: 
            state_sequence(Tensor(batch_size, 2 * horizon + 1, state_dim)): Previous H, now, and future H,
            target_sequence(Tensor(batch_size, 2 * horizon + 1, target_dim)): Previous H, now, and future H.
        """

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.train()

        # Rollout and Compute loss
        state = state_sequence[:, self.horizon, :]
        previous_state = state_sequence[:, :self.horizon, :]
        target = target_sequence[:, self.horizon, :]
        previous_target = target_sequence[:, :self.horizon, :]

        pos_loss, vel_loss = 0, 0
        stable_loss = 0
        time_step_metrics = {"pos_avg": [], "vel_avg": [],
                             "pos_1": [], "pos_2": [], "pos_3": [], "pos_4": [],
                             "vel_1": [], "vel_2": [], "vel_3": [], "vel_4": []}
        for k in range(self.train_horizon):
            next_target = target_sequence[:, self.horizon + k + 1, :]

            next_state_predict, delta_state_predict = self.predict(state, previous_state, target, previous_target, next_target)
            next_state = state_sequence[:, self.horizon + k + 1, :]
            p_loss, v_loss, p_mae, v_mae = self.compute_loss(next_state_predict, next_state)  # dim and batch mean
            pos_loss = pos_loss + (self.discount ** k) * p_loss
            vel_loss = vel_loss + (self.discount ** k) * v_loss

            if epoch_end:
                time_step_metrics["pos_1"].append(float(p_mae[0].item()))
                time_step_metrics["pos_2"].append(float(p_mae[1].item()))
                time_step_metrics["pos_3"].append(float(p_mae[2].item()))
                time_step_metrics["pos_4"].append(float(p_mae[3].item()))
                time_step_metrics["pos_avg"].append(float(torch.mean(p_mae).item()))
                time_step_metrics["vel_1"].append(float(v_mae[0].item()))
                time_step_metrics["vel_2"].append(float(v_mae[1].item()))
                time_step_metrics["vel_3"].append(float(v_mae[2].item()))
                time_step_metrics["vel_4"].append(float(v_mae[3].item()))
                time_step_metrics["vel_avg"].append(float(torch.mean(v_mae).item()))

            if self.k_stable != 0:
                assert base_predictor is not None
                _, delta_state_predict_target = (base_predictor.predict(
                    state, previous_state, target, previous_target, next_target))
                stb_loss = F.mse_loss(delta_state_predict, delta_state_predict_target.detach())  # dim and batch mean
                stable_loss = stable_loss + stb_loss * self.k_stable

            previous_state = self.sequence_update(previous_state, state)  # make sure no inplace operation
            previous_target = self.sequence_update(previous_target, target)
            state = next_state_predict
            target = next_target

        pos_loss_avg = pos_loss / self.train_horizon  # horizon mean
        vel_loss_avg = vel_loss / self.train_horizon  # horizon mean
        stable_loss = stable_loss / self.train_horizon
        total_loss = pos_loss_avg * self.k_pos + vel_loss_avg * (1 - self.k_pos) + stable_loss

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        self.optim.step()

        # Return eval statistics
        self.eval()
        self.iteration += 1

        iter_metrics = {"pos_loss": float(pos_loss_avg.item()),
                        "vel_loss": float(vel_loss_avg.item()),
                        "total_loss": float(total_loss.item()),
                        "grad_norm": float(grad_norm),
                        "data_num": container.num_push}

        return epoch_end, iter_metrics, time_step_metrics

