import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

device = torch.device('cuda')


parser = argparse.ArgumentParser()

parser.add_argument('--gamma', default=0.98, type=int)  # discount gamma 0.98
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient 0.005
parser.add_argument('--buffer_size', default=100000, type=int)   # replay buffer size
parser.add_argument('--batch_size', default=4096, type=int)  # mini batch size
parser.add_argument('--update_time', default=50, type=int)
parser.add_argument('--learning_rate', default=3e-5, type=int)

args = parser.parse_args()


class Replay_buffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.capacity = buffer_size
        self.num_push = 0

        self.state_pool = torch.zeros(self.capacity, state_dim).float().to(device)
        self.action_pool = torch.zeros(self.capacity, action_dim).float().to(device)
        self.reward_pool = torch.zeros(self.capacity, 1).float().to(device)
        self.next_state_pool = torch.zeros(self.capacity, state_dim).float().to(device)
        self.done_pool = torch.zeros(self.capacity, 1).int().to(device)

    def push(self, s, a, s_, r, d):
        index = self.num_push % self.capacity

        s = torch.tensor(s).float().to(device)
        a = torch.tensor(a).float().to(device)
        r = torch.tensor(r).float().to(device)
        s_ = torch.tensor(s_).float().to(device)
        d = torch.tensor(d).int().to(device)

        for pool, ele in zip([self.state_pool, self.action_pool, self.reward_pool,
                              self.next_state_pool, self.done_pool], [s, a, r, s_, d]):
            pool[index] = ele

        self.num_push += 1

    def sample(self, batch_size):
        if self.num_push > batch_size:
            if self.num_push > self.capacity:
                index = np.random.choice(range(self.capacity), batch_size, replace=False)
            else:
                index = np.random.choice(range(self.num_push), batch_size, replace=False)
        else:
            index = np.random.choice(range(self.num_push), batch_size, replace=True)

        bn_s, bn_a, bn_r, bn_s_, bn_d = (self.state_pool[index], self.action_pool[index],
                                         self.reward_pool[index], self.next_state_pool[index], self.done_pool[index])
        return bn_s, bn_a, bn_r, bn_s_, bn_d


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        # self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        # self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512)
        # self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 128)
        # self.ln4 = nn.LayerNorm(128)
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
        # self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512)
        # self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 128)
        # self.ln4 = nn.LayerNorm(128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, s):
        # x = F.elu(self.ln1(self.fc1(s)))
        # x = F.elu(self.ln2(self.fc2(x)))
        # x = F.elu(self.ln3(self.fc3(x)))
        # x = F.elu(self.ln4(self.fc4(x)))
        x = F.relu(self.ln1(self.fc1(s)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        v = self.fc5(x)
        return v


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        # self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512)
        # self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 128)
        # self.ln4 = nn.LayerNorm(128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1)  # combination s and a

        # x = F.elu(self.ln1(self.fc1(x)))
        # x = F.elu(self.ln2(self.fc2(x)))
        # x = F.elu(self.ln3(self.fc3(x)))
        # x = F.elu(self.ln4(self.fc4(x)))
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        v = self.fc5(x)
        return v


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


class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = PolicyNet(state_dim, action_dim).to(device)
        self.value_net = ValueNet(state_dim).to(device)
        self.policy_net.apply(weight_init)
        self.value_net.apply(weight_init)

        self.soft_value_net = ValueNet(state_dim).to(device)  # soft update
        for soft_param, param in zip(self.soft_value_net.parameters(), self.value_net.parameters()):
            soft_param.data.copy_(param.data)

        self.q_net1 = QNet(state_dim, action_dim).to(device)
        self.q_net2 = QNet(state_dim, action_dim).to(device)
        self.q_net1.apply(weight_init)
        self.q_net2.apply(weight_init)

        self.replay_buffer = Replay_buffer(args.buffer_size, state_dim, action_dim)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=args.learning_rate)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=args.learning_rate)

        self.MSE_criterion = nn.MSELoss()

        self.writer = SummaryWriter(os.path.dirname(__file__) + '/SAC_exps/SAC_exps_new')

        self.num_update = 0

    def train(self, state):
        state = torch.tensor(state).float().to(device)  # transport float(from env) to tensor
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)

        action = Normal(mu, sigma).sample()
        # using tanh to restrict output in (-1, 1)
        action_out = torch.tanh(action).detach().cpu().numpy().tolist()

        return action_out

    def test(self, state):
        state = torch.tensor(state).float().to(device)  # transport float(from env) to tensor
        mu, _ = self.policy_net(state)

        action = mu
        action_out = torch.tanh(action).detach().cpu().numpy().tolist()

        return action_out

    def evaluate(self, state):
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        noise = Normal(0, 1)
        z = noise.sample()
        action = mu + sigma * z.to(device)
        sample_action = torch.tanh(action)

        log_prob = dist.log_prob(action) - (2 * (np.log(2) - action - F.softplus(-2 * action)))  # correction of tanh
        log_prob = log_prob.sum(-1).reshape(args.batch_size, 1)

        return sample_action, log_prob

    def update(self, epoch):
        if self.num_update % 500 == 0:
            print("Training ... \t{} times ".format(self.num_update))

        for _ in range(args.update_time):
            bn_s, bn_a, bn_r, bn_s_, bn_d = self.replay_buffer.sample(args.batch_size)

            # compute q loss
            predict_next_value = self.soft_value_net(bn_s_)  # soft value to soft q
            target_q = bn_r + (1 - bn_d) * args.gamma * predict_next_value

            predict_q1 = self.q_net1(bn_s, bn_a)
            predict_q2 = self.q_net2(bn_s, bn_a)

            q1_loss = self.MSE_criterion(predict_q1, target_q.detach()).mean()
            q2_loss = self.MSE_criterion(predict_q2, target_q.detach()).mean()

            # compute policy loss
            sample_action, log_prob = self.evaluate(bn_s)

            excepted_new_Q = torch.min(self.q_net1(bn_s, sample_action), self.q_net2(bn_s, sample_action))
            policy_loss = (log_prob - excepted_new_Q).mean()  # max entropy and q value

            # compute value loss
            target_value = excepted_new_Q - log_prob  # soft value
            predict_value = self.value_net(bn_s)
            value_loss = self.MSE_criterion(predict_value, target_value.detach()).mean()

            self.writer.add_scalar('Loss/value_loss', value_loss, global_step=self.num_update)
            self.writer.add_scalar('Loss/q1_loss', q1_loss, global_step=self.num_update)
            self.writer.add_scalar('Loss/q2_loss', q2_loss, global_step=self.num_update)
            self.writer.add_scalar('Loss/policy_loss', policy_loss, global_step=self.num_update)
            self.writer.add_scalar('Data/data_num', self.replay_buffer.num_push, global_step=self.num_update)
            self.writer.add_scalar('Data/epoch', epoch, global_step=self.num_update)

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            self.q1_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()

            value_loss.backward()
            q1_loss.backward()
            q2_loss.backward()
            policy_loss.backward()

            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.q_net1.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.q_net2.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)

            self.value_optimizer.step()
            self.q1_optimizer.step()
            self.q2_optimizer.step()
            self.policy_optimizer.step()

            # value net soft update
            for soft_param, param in zip(self.soft_value_net.parameters(), self.value_net.parameters()):
                soft_param.data.copy_(soft_param * (1 - args.tau) + param * args.tau)

            self.num_update += 1

    def save(self, epoch):
        os.makedirs(os.path.dirname(__file__) + '/SAC_params/policy_net/', exist_ok=True)
        os.makedirs(os.path.dirname(__file__) + '/SAC_params/value_net/', exist_ok=True)
        os.makedirs(os.path.dirname(__file__) + '/SAC_params/q_net1/', exist_ok=True)
        os.makedirs(os.path.dirname(__file__) + '/SAC_params/q_net2/', exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.dirname(__file__) +
                   '/SAC_params/policy_net/' + str(epoch) + '.pth')
        torch.save(self.value_net.state_dict(), os.path.dirname(__file__) +
                   '/SAC_params/value_net/' + str(epoch) + '.pth')
        torch.save(self.q_net1.state_dict(), os.path.dirname(__file__) +
                   '/SAC_params/q_net1/' + str(epoch) + '.pth')
        torch.save(self.q_net2.state_dict(), os.path.dirname(__file__) +
                   '/SAC_params/q_net2/' + str(epoch) + '.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, epoch, name=None):
        if name is None:
            dirr = os.path.dirname(__file__) + '/SAC_params'
        else:
            dirr = os.path.dirname(__file__) + '/SAC_params_' + name
        
        self.policy_net.load_state_dict(torch.load(dirr + '/policy_net/' + str(epoch) + '.pth'))
        self.value_net.load_state_dict(torch.load(dirr + '/value_net/' + str(epoch) + '.pth'))
        self.q_net1.load_state_dict(torch.load(dirr + '/q_net1/' + str(epoch) + '.pth'))
        self.q_net2.load_state_dict(torch.load(dirr + '/q_net2/' + str(epoch) + '.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")