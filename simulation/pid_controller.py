import numpy as np


class PidController:
    def __init__(self):
        self.history_error = np.array([0, 0, 0, 0])

    def pid_control(self, target_pos, now_pos):
        # pwm_lower_bounds = np.array([-950, -800, -800, -700])  # joy range
        # pwm_upper_bounds = np.array([800, 800, 800, 700])
        Ku = np.array([4000, 1400, 100, 3000])  # ZN method
        Tu = np.array([1.23, 1.28, 1.26, 2]) * 500  # 500

        kd = 0.1 * Ku * Tu * 0.08
        kp = np.array([3800, 2400, 400, 2400]) * 0.55
        # kd = 0 * kd

        direction_boom_arm_bucket = (target_pos - now_pos) < 0
        direction_boom_arm_bucket = direction_boom_arm_bucket[:3]
        delta_swing = target_pos[3:] - now_pos[3:]
        direction_swing = delta_swing > 0
        index_wc = np.where(np.abs(delta_swing) > np.pi)      ###3---------------------------------------------------------------
        direction_swing[index_wc] = ~direction_swing[index_wc]
        direction = np.concatenate([direction_boom_arm_bucket, direction_swing], axis=-1)
        direction = direction.astype(np.float32) * 2 - 1

        current_complex = np.exp(1j * now_pos)
        first_targetpos_complex = np.exp(1j * target_pos)
        current_error = np.abs(np.angle(current_complex / first_targetpos_complex)) * direction

        action_ini = direction * 0  # 400
        # current_error = np.abs(target_pos - now_pos) * direction
        delta_error = (current_error - self.history_error) / (0.05 * 1000)
        pid_action = action_ini + kp * current_error + kd * delta_error
        # pid_action = np.clip(pid_action, pwm_lower_bounds, pwm_upper_bounds)

        self.history_error = current_error
        return pid_action
