import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from network_model.network import mlp, mlp_norm
from network_model.weight_init import weight_init
from network_model.space_change import sincos_to_rad_for_pos
from network_model.angle_error import angle_error
from network_model.predictor import Predictor
from offline_train.container import Container

from online_train.buffer import Buffer


class Controller(nn.Module):
    """
    Controller model architecture.
    """

    def __init__(self, cfg):
        super().__init__()        
        self.device = torch.device(cfg.device)
        self.horizon = cfg.horizon
        self.train_horizon = cfg.train_horizon
        self.discount = cfg.controller_discount
        self.represent = cfg.represent
        self.frequency = cfg.frequency
        #   self.threshold = cfg.controller_change_threshold
        self.k_smooth = cfg.controller_smooth_k
        self.max_range = cfg.controller_max_range
        self.k_stable = cfg.controller_stable_k

        self.iteration = 0

        controller_dim = (cfg.horizon + 1) * cfg.state_dim + (2 * cfg.horizon + 1) * cfg.target_dim
        if cfg.controller_type == 'mlp':
            self._controller = mlp(controller_dim, cfg.controller_hidden_depth * [cfg.controller_hidden_dim],
                                   cfg.target_dim, tanh_out=True)
        elif cfg.controller_type == 'mlp_norm':
            self._controller = mlp_norm(controller_dim, cfg.controller_hidden_depth * [cfg.controller_hidden_dim],
                                        cfg.target_dim, cfg.controller_dropout, tanh_out=True)
        else:
            raise ValueError('Unknown predictor type: {}'.format(cfg.controller_type))
        self.apply(weight_init)
        self._controller = self._controller.to(self.device)

        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.controller_lr)
        self.scheduler = None
        if cfg.controller_T_max > 0:
            self.scheduler = CosineAnnealingLR(self.optim, T_max=cfg.controller_T_max)

        self.max_norm = cfg.controller_max_norm

        self.eval()

    @property
    def total_params(self):
        """
        Total params number of params requires grad.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def control(self, state, previous_state, target, previous_target, future_real_target):
        """
        Predicts the next target given

        Args:
            state(Tensor(batch_size, state_dim)),
            previous_state(Tensor(batch_size, horizon, state_dim)),
            target(Tensor(batch_size, target_dim)),
            previous_target(Tensor(batch_size, horizon, target_dim)),
            future_real_target(Tensor(batch_size, horizon, target_dim)).
        
        Returns:
            next_target(Tensor(batch_size, target_dim)),
            delta_target(Tensor(batch_size, target_dim)),
            avg_delta_target(Tensor(target_dim)).
        """
        next_real_target = future_real_target[:, 0, :]

        previous_state = previous_state.view(previous_state.shape[0], -1)
        previous_target = previous_target.view(previous_target.shape[0], -1)
        future_real_target = future_real_target.reshape(future_real_target.shape[0], -1)

        x = torch.cat([previous_state, state, previous_target, target, future_real_target], dim=-1)
        # delta_target = self.max_range * self._controller(x)
        delta_target = self._controller(x) * (torch.tensor([0.15, 0.3, 0.4, 0.4]).to(self.device))
        next_target = next_real_target + delta_target
        # next_target = self._controller(x)
        # delta_target = next_target

        return next_target, delta_target, torch.mean(delta_target.detach(), dim=0)  # batch mean

    def online_control(self, rl_state):
        x = torch.tensor(rl_state, dtype=torch.float32).to(device=self.device)
        next_target = self._controller(x)

        return next_target

    def save(self, fp):
        """
        Save state dict of the agent to filepath.
        
        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"controller": self.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.
        
        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, weights_only=True)
        self.load_state_dict(state_dict["controller"])

    def compute_loss(self, next_state, next_real_target):
        """
        Compute loss in rad space between predicted and actual next state.

        Args:
            next_state (torch.Tensor): Predicted next state (of shape (batch_size, state_dim)),
            next_real_target (torch.Tensor): Actual next target (of shape (batch_size, target_dim)).

        Returns:
            torch.Tensor: Mean squared error loss between predicted and actual next state.
        """
        if self.represent == 'rad':
            next_pos = next_state[:, :4]
        elif (self.represent == 'sin-cos') or (self.represent == 'sin-cos-special-v'):
            next_pos = sincos_to_rad_for_pos(next_state)
            next_real_target = sincos_to_rad_for_pos(next_real_target)
        else:
            raise ValueError(f"Invalid represent: {self.represent}")

        pos_error = angle_error(next_pos, next_real_target)
        loss = F.mse_loss(pos_error, torch.zeros_like(pos_error))  # dim and batch mean

        mae = torch.mean(torch.abs(pos_error.detach()), dim=0)  # batch mean only

        return loss, mae
    
    def sequence_update(self, previous, new):
        """
        Update previous_state, previous_target and future_real_target.

        Args:
            previous (Tensor(batch_size, horizon, state_dim/target_dim)),
            new (Tensor(batch_size, state_dim/target_dim)).

        Returns:
            updated (Tensor(batch_size, horizon, state_dim/target_dim)):
                previous_state/target/future_real_target for next step.
        """

        updated = torch.empty_like(previous)
        updated[:, :-1, :] = previous[:, 1:, :].clone()
        updated[:, -1, :] = new.clone()
        return updated
    
    def update(self, container: Container or Buffer, predictor: Predictor, base_controller=None):
        """
        Main update function. Corresponds to one iteration of policy learning. 
        
        Args:
            container (data_process.container.Container): Data container.
            predictor (Predictor): Model to predict next state.
            base_controller (Controller): Policy to keep near to.
        
        Returns:
            dict: Dictionary of training statistics.
        """

        state_sequence, target_sequence, real_target_sequence, epoch_end = container.sample()
        state_sequence = state_sequence.to(self.device).requires_grad_(False)
        target_sequence = target_sequence.to(self.device).requires_grad_(False)
        real_target_sequence = real_target_sequence.to(self.device).requires_grad_(False)
        """
        None grad data: 
            state_sequence(Tensor(batch_size, 2 * horizon + 1, state_dim)): Previous H, now, and future H,
            target_sequence(Tensor(batch_size, 3 * horizon, target_dim)): Previous H, now, and future 2*H - 1,
            real_target_sequence(Tensor(batch_size, 3 * horizon, target_dim)): Previous H, now, and future 2*H - 1.
        """

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.train()
        for param in predictor.parameters():
            param.requires_grad = False

        # Rollout and Compute tracking error
        state = state_sequence[:, self.horizon, :]
        previous_state = state_sequence[:, :self.horizon, :]
        target = target_sequence[:, self.horizon, :]
        previous_target = target_sequence[:, :self.horizon, :]
        future_real_target = real_target_sequence[:, self.horizon:(self.horizon * 2), :]

        track_loss = 0
        time_step_metrics = {"dt_avg": [], "mae_avg": [],
                             "dt_1": [], "dt_2": [], "dt_3": [], "dt_4": [],
                             "mae_1": [], "mae_2": [], "mae_3": [], "mae_4": []}
        smooth_loss = 0
        stable_loss = 0
        last_delta_target = torch.zeros_like(target)
        for k in range(self.train_horizon):
            future_real_target = self.sequence_update(future_real_target, real_target_sequence[:, self.horizon * 2 + k, :])

            next_target, delta_target, avg_delta_target = self.control(
	            state, previous_state, target, previous_target, future_real_target)
            next_state, _ = predictor.predict(state, previous_state, target, previous_target, next_target)

            if epoch_end:
                time_step_metrics["dt_1"].append(float(avg_delta_target[0].item()))
                time_step_metrics["dt_2"].append(float(avg_delta_target[1].item()))
                time_step_metrics["dt_3"].append(float(avg_delta_target[2].item()))
                time_step_metrics["dt_4"].append(float(avg_delta_target[3].item()))
                time_step_metrics["dt_avg"].append(float(torch.mean(avg_delta_target).item()))

            next_real_target = future_real_target[:, 0, :]
            loss, mae = self.compute_loss(next_state, next_real_target)  # dim and batch mean
            track_loss = track_loss + (self.discount ** k) * loss

            # smooth loss
            # max_delta = (delta_target - last_delta_target).max().item()
            if k > 0:  # and (max_delta > self.threshold):
                smt_loss = F.mse_loss(delta_target, last_delta_target)  # dim and batch mean
                smooth_loss = smooth_loss + smt_loss * self.k_smooth
            last_delta_target = delta_target  # no detach version
            # last_delta_target = delta_target.detach()  # detach version

            if self.k_stable != 0:
                assert base_controller is not None
                _, base_delta_target, _ = (base_controller.control(
                    state, previous_state, target, previous_target, future_real_target))
                stb_loss = F.mse_loss(delta_target, base_delta_target.detach())  # dim and batch mean
                stable_loss = stable_loss + stb_loss * self.k_stable
            time_step_metrics["mae_1"].append(float(mae[0].item()))
            time_step_metrics["mae_2"].append(float(mae[1].item()))
            time_step_metrics["mae_3"].append(float(mae[2].item()))
            time_step_metrics["mae_4"].append(float(mae[3].item()))
            time_step_metrics["mae_avg"].append(float(torch.mean(mae).item()))

            previous_state = self.sequence_update(previous_state, state)  # make sure no inplace operation
            previous_target = self.sequence_update(previous_target, target)
            state = next_state
            target = next_target

        track_loss = track_loss / self.train_horizon  # horizon mean
        smooth_loss = smooth_loss / self.train_horizon
        stable_loss = stable_loss / self.train_horizon
        total_loss = track_loss + smooth_loss + stable_loss

        # Update model
        total_loss.backward()
        if self.max_norm != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        else:
            grad_norm = 0
        self.optim.step()
        if self.scheduler:
            self.scheduler.step()

        # Return eval statistics
        self.eval()
        for param in predictor.parameters():
            param.requires_grad = True
        self.iteration += 1

        iter_metrics = {"tracking_loss": float(track_loss.item()),
                        "smooth_loss": float(smooth_loss.item()),
                        "total_loss": float(total_loss.item()),
                        "grad_norm": float(grad_norm),
                        "data_num": container.num_push}
        if self.scheduler:
            iter_metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        return epoch_end, iter_metrics, time_step_metrics

