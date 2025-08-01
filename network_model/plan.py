import torch
# @torch.no_grad()
#     def plan(state, previous_state, target, previous_target,
#              future_real_target, predictor: Predictor, t0=False, eval_mode=True):
#         """
#         Plan a sequence of actions using the learned world model.
#         Adapted from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py .
#
#         Args:
#             state(Tensor(1, state_dim)),
#             previous_state(Tensor(1, horizon, state_dim)),
#             target(Tensor(1, target_dim)),
#             previous_target(Tensor(1, horizon, target_dim)),
#             future_real_target(Tensor(1, horizon, target_dim)).
#
#             predictor(Predictor): World model to use for planning.
#             t0 (bool): Whether this is the first observation in the episode.
#             eval_mode (bool): Whether to use the mean of the action distribution.
#
#         Returns:
#             torch.Tensor: Action to take in the environment.
#         """
#
#         # Initialize state and parameters
#         action_dim = target.size(1)
#
#         state = state.repeat(self.plan_num_samples, 1)
#         previous_state = previous_state.repeat(self.plan_num_samples, 1, 1)
#         target = target.repeat(self.plan_num_samples, 1)
#         previous_target = previous_target.repeat(self.plan_num_samples, 1, 1)
#         future_real_target = future_real_target.repeat(self.plan_num_samples, 1, 1)
#
#         mean = torch.zeros(self.horizon, action_dim, device=self.device)
#         std = self.plan_max_std * torch.ones(self.horizon, action_dim, device=self.device)
#         if not t0:
#             mean[:-1] = self._prev_mean[1:]
#
#         # Iterate MPPI
#         for _ in range(self.plan_iters):
#
#             # Sample actions
#             actions = (mean.unsqueeze(1) + std.unsqueeze(1) *
#                        torch.randn(self.horizon, self.plan_num_samples, action_dim,
#                                    device=self.device)).clamp(-self.max_range, self.max_range)
#
#             # Compute elite actions
#             loss = self.rollout(state, previous_state, target, previous_target, future_real_target, actions, predictor)
#             elite_idxs = torch.topk(- loss.squeeze(1), self.plan_num_elites, dim=0).indices
#             elite_loss, elite_actions = loss[elite_idxs], actions[:, elite_idxs]
#
#             # Update parameters
#             min_loss = elite_loss.min(0)[0]
#             score = torch.exp(self.plan_temperature * (min_loss - elite_loss))
#             score /= score.sum(0)
#             mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
#             std = torch.sqrt(
#                 torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1)
#                 / (score.sum(0) + 1e-9)).clamp_(self.plan_min_std, self.plan_max_std)
#
#         # Select action
#         score = score.squeeze(1).cpu().numpy()
#         actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
#         self._prev_mean = mean
#         a, std = actions[0], std[0]
#         if not eval_mode:
#             a += std * torch.randn(self.cfg.action_dim, device=std.device)
#         return a.clamp_(-self.max_range, self.max_range)