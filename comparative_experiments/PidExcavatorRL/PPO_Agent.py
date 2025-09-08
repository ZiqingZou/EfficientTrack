import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import os

device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', default=0.98, type=int)  # discount rate
parser.add_argument('--lamda', default=0.99, type=int)  # advantage discount
parser.add_argument('--buffer_size', default=60000, type=int)  # T step * N agent (N = 10)
parser.add_argument('--batch_size', default=4096, type=int)  # <=buffer.push_num
parser.add_argument('--update_time', default=8, type=int)  # params update time for one epoch
parser.add_argument('--clip_epsilon', default=3, type=float)  # clipped surrogate hyperparams
parser.add_argument('--learning_rate_policy', default=3e-5, type=float)  # for Adam optimizer 3e-4
parser.add_argument('--learning_rate_value', default=1e-6, type=float)  # for Adam optimizer 3e-4
parser.add_argument('--weight_decay_value', default=1e-5, type=float)  # for Adam optimizer 1e-3
args = parser.parse_args()


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512)
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 128)
        self.ln4 = nn.LayerNorm(128)
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)

    def forward(self, s):
        # x = F.elu(self.ln1(self.fc1(s)))
        # x = F.elu(self.ln2(self.fc2(x)))
        # x = F.elu(self.ln3(self.fc3(x)))
        # x = F.elu(self.ln4(self.fc4(x)))
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)

        # give a restriction on the chosen action
        log_std = torch.clamp(log_std, -10, 2)

        return mu, log_std


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512)
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 128)
        self.ln4 = nn.LayerNorm(128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, s):
        x = F.relu(self.ln1(self.fc1(s)))
        # x = F.elu(self.ln2(self.fc2(x)))
        # x = F.elu(self.ln3(self.fc3(x)))
        # x = F.elu(self.ln4(self.fc4(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        v = self.fc5(x)
        return v


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.capacity = buffer_size
        self.num_push = 0  # data number

        self.state_pool = torch.zeros(self.capacity, state_dim).float().to(device)
        self.action_pool = torch.zeros(self.capacity, action_dim).float().to(device)
        self.next_state_pool = torch.zeros(self.capacity, state_dim).float().to(device)
        self.reward_pool = torch.zeros(self.capacity, 1).float().to(device)
        self.done_pool = torch.zeros(self.capacity, 1).int().to(device)

        self.advantage_pool = torch.zeros(self.capacity, 1).float().to(device)
        self.log_prob_old_pool = torch.zeros(self.capacity, 1).float().to(device)
        self.td_target_pool = torch.zeros(self.capacity, 1).float().to(device)

        self.start_index = []  # start element index of one agent
        self.end_index = []  # end element index of one agent

    def push(self, s, a, s_, r, d, start, end):
        index = self.num_push
        if start:
            self.start_index.append(index)
        if end:
            self.end_index.append(index)

        s = torch.tensor(s).float().to(device)
        a = torch.tensor(a).float().to(device)
        s_ = torch.tensor(s_).float().to(device)
        r = torch.tensor(r).float().to(device)
        d = torch.tensor(d).int().to(device)

        for pool, ele in zip([self.state_pool, self.action_pool, self.next_state_pool,
                              self.reward_pool, self.done_pool], [s, a, s_, r, d]):
            pool[index] = ele

        self.num_push += 1

    def sample(self, batch_size):
        if self.num_push > batch_size:
            index = np.random.choice(range(self.num_push), batch_size, replace=False)
        else:
            index = np.random.choice(range(self.num_push), batch_size, replace=True)

        bn_s = self.state_pool[index]
        bn_a = self.action_pool[index]
        bn_s_ = self.next_state_pool[index]
        bn_r = self.reward_pool[index]
        bn_d = self.done_pool[index]

        bn_ad = self.advantage_pool[index]
        bn_lpo = self.log_prob_old_pool[index]
        bn_tdt = self.td_target_pool[index]

        return bn_s, bn_a, bn_s_, bn_r, bn_d, bn_ad, bn_lpo, bn_tdt

    def compute(self, value_net, policy_net):
        # for all elements in buffer
        next_value_predict = value_net(self.next_state_pool)
        td_target = self.reward_pool + args.gamma * next_value_predict * (1 - self.done_pool)
        self.td_target_pool = td_target.detach()

        td_value = value_net(self.state_pool)
        td_delta = td_target - td_value
        td_delta = td_delta.cpu().detach().numpy()

        # compute advantage
        advantage_list = []
        for agent_i in range(len(self.start_index)):
            advantage = 0
            ad_list = []
            td_d = td_delta[self.start_index[agent_i]:(self.end_index[agent_i]+1)]
            for delta in td_d[::-1]:
                advantage = args.gamma * args.lamda * advantage + delta
                ad_list.append(advantage)
            ad_list.reverse()
            advantage_list.extend(ad_list)
        # self.advantage_pool = torch.tensor(advantage_list).float().to(device).detach()
        advantage_array = np.array(advantage_list)
        self.advantage_pool = torch.from_numpy(advantage_array).float().to(device).detach()

        # compute log(pi_old)
        mu, log_sigma = policy_net(self.state_pool)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu.detach(), sigma.detach())
        action = torch.clamp(torch.atanh(self.action_pool), -5, 5)
        log_prob = dist.log_prob(action) - (2 * (np.log(2) - action - F.softplus(-2 * action)))  # correction of tanh
        log_prob = log_prob.clamp(min=-20).sum(-1).reshape(self.capacity, 1)

        self.log_prob_old_pool = log_prob.detach()


def weight_init(m):
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                nn.init.trunc_normal_(p, std=0.02)  # Weight
                nn.init.constant_(m[i+1], 0)  # Bias


class PPOAgent:
    def __init__(self, state_dim=9, action_dim=2):
        self.policy_net = PolicyNet(state_dim, action_dim).to(device)
        self.value_net = ValueNet(state_dim).to(device)
        self.policy_net.apply(weight_init)
        self.value_net.apply(weight_init)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(),
                                          lr=args.learning_rate_value, weight_decay=args.weight_decay_value)

        self.MSE_criterion = nn.MSELoss()

        self.writer = SummaryWriter(os.path.dirname(__file__) + '/PPO_exps/PPO_exps_new')

        self.num_update = 0  # params update time
        self.total_data_num = 0

    def train(self, state):
        state = torch.tensor(state).float().to(device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)

        action = Normal(mu, sigma).sample()
        # using tanh to restrict output in (-1, 1)
        action_out = torch.tanh(action).detach().cpu().numpy().tolist()

        return action_out

    def test(self, state):
        state = torch.tensor(state).float().to(device)
        mu, _ = self.policy_net(state)

        action = mu
        action_out = torch.tanh(action).detach().cpu().numpy().tolist()

        return action_out

    def predict(self, state):
        state = torch.tensor(state).float().to(device)
        value = self.value_net(state)
        value_out = value.detach().cpu().numpy()

        return value_out

    def update(self, replay_buffer: ReplayBuffer):
        replay_buffer.compute(self.value_net, self.policy_net)
        self.total_data_num += replay_buffer.num_push

        if self.num_update % 500 == 0:
            print("Training ... \t{} times ".format(self.num_update))
        for _ in range(args.update_time):

            bn_s, bn_a, bn_s_, bn_r, bn_d, bn_ad, bn_lpo, bn_tdt = replay_buffer.sample(args.batch_size)

            # compute policy loss
            mu, log_sigma = self.policy_net(bn_s)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            action = torch.clamp(torch.atanh(bn_a), -5, 5)
            log_prob = dist.log_prob(action) - (
                        2 * (np.log(2) - action - F.softplus(-2 * action)))  # correction of tanh
            log_prob = log_prob.sum(-1).reshape(args.batch_size, 1)

            ratio = torch.exp(log_prob - bn_lpo.detach())
            sur1 = ratio * (bn_ad.detach())
            sur2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * (bn_ad.detach())
            policy_loss = torch.mean(-torch.min(sur1, sur2))

            print('policy_loss:', float(policy_loss))
            self.writer.add_scalar('Loss/policy_loss', policy_loss, global_step=self.num_update)

            # compute value loss
            predict_v = self.value_net(bn_s)
            value_loss = self.MSE_criterion(predict_v, bn_tdt.detach()).mean()

            print('value_loss:', float(value_loss))
            self.writer.add_scalar('Loss/value_loss', value_loss, global_step=self.num_update)

            self.writer.add_scalar('Data/data_num', self.total_data_num, global_step=self.num_update)

            # mini batch gradient descent
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()

            self.num_update += 1

    def save(self, epoch):
        os.makedirs(os.path.dirname(__file__) + '/PPO_params/policy_net/', exist_ok=True)
        os.makedirs(os.path.dirname(__file__) + '/PPO_params/value_net/', exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.dirname(__file__) +
                   '/PPO_params/policy_net/' + str(epoch) + '.pth')
        torch.save(self.value_net.state_dict(), os.path.dirname(__file__) +
                   '/PPO_params/value_net/' + str(epoch) + '.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, epoch, name=None):
        if name is None:
            fp = os.path.dirname(__file__) + '/PPO_params'
        else:
            fp = os.path.dirname(__file__) + '/PPO_params_' + name
        self.policy_net.load_state_dict(torch.load(fp + '/policy_net/' + str(epoch) + '.pth'))
        self.value_net.load_state_dict(torch.load(fp + '/value_net/' + str(epoch) + '.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
