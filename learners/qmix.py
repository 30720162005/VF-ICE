from networks.agent import rnn_agent
from networks.value_based import qmixer,qmixer_ice
import torch
import numpy as np
import random
from collections import deque

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class EpsilonGreedy:
    def __init__(self, action_nb, agent_nb, final_step, epsilon_start=float(1), epsilon_end=0.05):
        self.epsilon = epsilon_start
        self.initial_epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.action_nb = action_nb
        self.final_step = final_step
        self.agent_nb = agent_nb

    def act(self, value_action, avail_actions):
        # 贪婪算法选择动作
        if np.random.random() > self.epsilon:
            action = value_action.max(dim=1)[
                1].cpu().detach().numpy()  # 查找每行的最大值并返回列下标 dim=0，每列最大值；dim=1，每行最大值；不备注，所有最大值
        else:
            action = torch.distributions.Categorical(avail_actions).sample().long().cpu().detach().numpy()
        return action

    def epislon_decay(self, step):
        # 得到epsilon的值，if < 0.05,则为0.05，否则为1-progress
        progress = step / self.final_step

        decay = self.initial_epsilon - progress
        if decay <= self.epsilon_end:
            decay = self.epsilon_end
        self.epsilon = decay


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, t, obs, available_actions, filled):
        # 把数据存入ReplayBuffer池中
        experience = [s, a, r, t, obs, available_actions, np.array([filled])]
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        # 采取前batch_size个样本数据
        batch = []
        for idx in range(batch_size):
            batch.append(self.buffer[idx])
        batch = np.array(batch)

        s_batch = np.array([_[0] for _ in batch], dtype='float32')
        a_batch = np.array([_[1] for _ in batch], dtype='float32')
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        obs_batch = np.array([_[4] for _ in batch], dtype='float32')
        available_actions_batch = np.array([_[5] for _ in batch], dtype='float32')
        filled_batch = np.array([_[6] for _ in batch], dtype='float32')

        return s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch

    def clear(self):
        # 清除Replaybuffer中的所有数据
        self.buffer.clear()
        self.count = 0


class EpisodeBatch:
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()  # 双头队列

    def reset(self):
        pass

    def add(self, replay_buffer):
        # 存储数据
        if self.count < self.buffer_size:
            self.buffer.append(replay_buffer)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(replay_buffer)

    def _get_max_episode_len(self, batch):
        # 计算一局中的最大迭代次数
        max_episode_len = 0

        for replay_buffer in batch:
            _, _, _, t, _, _, _ = replay_buffer.sample_batch(replay_buffer.size())
            for idx, t_idx in enumerate(t):
                if t_idx == True:
                    if idx > max_episode_len:
                        max_episode_len = idx + 1
                    break

        return max_episode_len

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        episode_len = self._get_max_episode_len(batch)
        s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch = [], [], [], [], [], [], []
        for replay_buffer in batch:
            s, a, r, t, obs, available_actions, filled = replay_buffer.sample_batch(episode_len)
            s_batch.append(s)  # 维度 episode_len*282
            a_batch.append(a)  # 维度 episode_len*6*1
            r_batch.append(r)  # 维度 episode_len*1
            t_batch.append(t)  # 维度 episode_len*1
            obs_batch.append(obs)  # 维度 episode_len*6*192
            available_actions_batch.append(available_actions)  # 维度 episode_len*6*30
            filled_batch.append(filled)  # 维度 episode_len*1

        filled_batch = np.array(filled_batch)
        r_batch = np.array(r_batch)
        t_batch = np.array(t_batch)
        a_batch = np.array(a_batch)
        obs_batch = np.array(obs_batch)
        available_actions_batch = np.array(available_actions_batch)

        return s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch, episode_len

        # return batch

    def size(self):
        return self.count


class QMix:
    def __init__(self, training, agent_nb, obs_shape, states_shape, action_n, lr, gamma=0.99, batch_size=16,
                 replay_buffer_size=5000, update_target_network=200, final_step=50000):  # 32
        self.training = training
        self.gamma = gamma  # 0.99
        self.batch_size = batch_size  # 16
        self.update_target_network = update_target_network  # 200
        self.hidden_states = None
        self.target_hidden_states = None
        self.agent_nb = agent_nb  # 智能体数量
        self.action_n = action_n  # 动作数
        self.state_shape = states_shape
        self.obs_shape = obs_shape

        self.epsilon_greedy = EpsilonGreedy(action_n, agent_nb, final_step)  # 参数初始化
        self.episode_batch = EpisodeBatch(replay_buffer_size)  # 参数初始化

        self.agents = rnn_agent.RNNAgent(obs_shape, n_actions=action_n).to(device)
        self.target_agents = rnn_agent.RNNAgent(obs_shape, n_actions=action_n).to(device)
        self.qmixer = qmixer.QMixer(agent_nb, states_shape, mixing_embed_dim=32).to(device)
        self.target_qmixer = qmixer.QMixer(agent_nb, states_shape, mixing_embed_dim=32).to(device)

        self.target_agents.update(self.agents)
        self.target_qmixer.update(self.qmixer)

        self.params = list(self.agents.parameters())
        self.params += self.qmixer.parameters()

        self.optimizer = torch.optim.RMSprop(params=self.params, lr=lr, alpha=0.99, eps=0.00001)

    def save_model(self, episode, global_step, filename: object) -> object:
        all_state = {'agents': self.agents.state_dict(), 'optimizer': self.optimizer.state_dict(), 'episodes': episode,
                     'global_step': global_step}
        torch.save(all_state, filename)

    def load_model(self, filename):
        all_state = torch.load(filename)
        self.agents.load_state_dict(all_state['agents'])
        self.agents.eval()
        self.optimizer.load_state_dict(all_state['optimizer'])
        return all_state['episodes'], all_state['global_step']

    def _init_hidden_states(self, batch_size):
        self.hidden_states = self.agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)
        self.target_hidden_states = self.target_agents.init_hidden().unsqueeze(0).expand(batch_size, self.agent_nb, -1)

    def decay_epsilon_greddy(self, global_steps):
        self.epsilon_greedy.epislon_decay(global_steps)

    def on_reset(self, batch_size):
        self._init_hidden_states(batch_size)

    def update_targets(self, episode):
        if episode % self.update_target_network == 0 and self.training:
            self.target_agents.update(self.agents)
            self.target_qmixer.update(self.qmixer)
            pass

    def train(self):
        # 计算误差
        if self.training and self.episode_batch.size() > self.batch_size:
            for _ in range(2):
                self._init_hidden_states(self.batch_size)
                s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch, filled_batch, episode_len = self.episode_batch.sample_batch(
                    self.batch_size)

                r_batch = r_batch[:, :-1]  # 获取所有行的除最后一列以外的所有列数据
                a_batch = a_batch[:, :-1]
                t_batch = t_batch[:, :-1]
                filled_batch = filled_batch[:, :-1]

                mask = (1 - filled_batch) * (1 - t_batch)

                r_batch = torch.FloatTensor(r_batch).to(device)
                t_batch = torch.FloatTensor(t_batch).to(device)
                mask = torch.FloatTensor(mask).to(device)

                a_batch = torch.LongTensor(a_batch).to(device)

                mac_out = []

                for t in range(episode_len):
                    obs = obs_batch[:, t]
                    obs = np.concatenate(obs, axis=0)
                    obs = torch.FloatTensor(obs).to(device)
                    agent_actions, self.hidden_states = self.agents(obs, self.hidden_states)
                    agent_actions = agent_actions.view(self.batch_size, self.agent_nb, -1)
                    mac_out.append(agent_actions)
                mac_out = torch.stack(mac_out, dim=1)

                chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=a_batch).squeeze(3)

                target_mac_out = []

                for t in range(episode_len):
                    obs = obs_batch[:, t]
                    obs = np.concatenate(obs, axis=0)
                    obs = torch.FloatTensor(obs).to(device)
                    agent_actions, self.target_hidden_states = self.target_agents(obs, self.target_hidden_states)
                    agent_actions = agent_actions.view(self.batch_size, self.agent_nb, -1)
                    target_mac_out.append(agent_actions)
                target_mac_out = torch.stack(target_mac_out[1:], dim=1)
                available_actions_batch = torch.Tensor(available_actions_batch).to(device)

                target_mac_out[available_actions_batch[:, 1:] == 0] = -9999999

                target_max_qvals = target_mac_out.max(dim=3)[0]  # 目标网络：在30个动作值中选择最大的Q值

                states = torch.FloatTensor(s_batch).to(device)
                # states2 = torch.FloatTensor(s2_batch).to(device)

                chosen_action_qvals = self.qmixer(chosen_action_qvals, states[:, :-1])
                target_max_qvals = self.target_qmixer(target_max_qvals, states[:, 1:])

                yi = r_batch + self.gamma * (1 - t_batch) * target_max_qvals

                td_error = (chosen_action_qvals - yi.detach())

                mask = mask.expand_as(td_error)

                masked_td_error = td_error * mask

                loss = (masked_td_error ** 2).sum() / mask.sum()

                print('loss:', loss)
                self.optimizer.zero_grad()  # 梯度归零
                loss.backward()  # 反向传播计算得到每个参数的梯度
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, 10)  # 梯度裁剪，防止产生梯度消失/爆炸
                self.optimizer.step()  # 利用梯度下降更新参数

            pass
        pass

    def act(self, batch, obs, agents_available_actions):
        # 返回选择动作的下标
        value_action, self.hidden_states = self.agents(obs, self.hidden_states)
        value_action[agents_available_actions == 0] = -1e10
        if self.training:
            value_action = self.epsilon_greedy.act(value_action, agents_available_actions)
        else:
            value_action = np.argmax(value_action.cpu().data.numpy(), -1)
        value_action = value_action.reshape(batch, self.agent_nb, -1)
        return value_action
