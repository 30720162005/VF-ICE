from learners import iql, vdn, qmix, qmix_ice
from envs.smac.env import StarCraft2Env, PCStarCraft2Env
import numpy as np
import torch
from multiprocessing import Process, Lock, Pipe, Value
from threading import Thread
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), torch.float32


def env_run(algo_name, scenario, id, child_conn, locker, replay_buffer_size):
    # 连接环境
    if algo_name == 'PCQMIX' or algo_name == 'PCVDN':
        env = PCStarCraft2Env(map_name=scenario, replay_dir="./replay/")
    else:
        env = StarCraft2Env(map_name=scenario, replay_dir="./replay/")
    # 获取环境信息
    env_info = env.get_env_info()

    action_n = env_info["n_actions"]  # 动作数
    agent_nb = env_info["n_agents"]  # 智能体数
    state_shape = env_info["state_shape"]  # 状态空间大小
    obs_shape = env_info["obs_shape"] + agent_nb + action_n

    agent_id_one_hot = OneHot(agent_nb)
    actions_one_hot = OneHot(action_n)

    agent_id_one_hot_array = []
    for agent_id in range(agent_nb):
        agent_id_one_hot_array.append(agent_id_one_hot.transform(torch.FloatTensor([agent_id])).cpu().detach().numpy())
    agent_id_one_hot_array = np.array(agent_id_one_hot_array)
    actions_one_hot_reset = torch.zeros_like(torch.empty(agent_nb, action_n))
    # 初始化
    state_zeros = np.zeros(state_shape)
    obs_zeros = np.zeros((agent_nb, obs_shape))
    actions_zeros = np.zeros([agent_nb, 1])
    reward_zeros = 0
    agents_available_actions_zeros = np.zeros((agent_nb, action_n))
    agents_available_actions_zeros[:, 0] = 1

    child_conn.send(id)

    while True:

        while True:
            data = child_conn.recv()
            if data == 'save':
                env.save_replay()
                child_conn.send('save ok.')
            elif data == 'close':
                env.close()
                exit()
            else:
                break
        # 环境重置
        locker.acquire()
        env.reset()
        locker.release()

        episode_reward = 0
        episode_step = 0

        obs = np.array(env.get_obs())
        obs = np.concatenate([obs, actions_one_hot_reset, agent_id_one_hot_array], axis=-1)
        state = np.array(env.get_state())
        terminated = False

        while not terminated:
            agents_available_actions = []
            for agent_id in range(agent_nb):
                agents_available_actions.append(env.get_avail_agent_actions(agent_id))

            child_conn.send(["actions", obs, agents_available_actions])
            actions = child_conn.recv()

            reward, terminated, _ = env.step(actions)  # 获取单步奖励

            agents_available_actions2 = []
            for agent_id in range(agent_nb):
                agents_available_actions2.append(env.get_avail_agent_actions(agent_id))

            obs2 = np.array(env.get_obs())
            actions_one_hot_agents = []
            for action in actions:
                actions_one_hot_agents.append(
                    actions_one_hot.transform(torch.FloatTensor(action)).cpu().detach().numpy())
            actions_one_hot_agents = np.array(actions_one_hot_agents)

            obs2 = np.concatenate([obs2, actions_one_hot_agents, agent_id_one_hot_array], axis=-1)
            state2 = np.array(env.get_state())

            child_conn.send(["replay_buffer", state, actions, [reward], [terminated], obs, agents_available_actions, 0])

            episode_reward += reward
            episode_step += 1

            obs = obs2
            state = state2

        for _ in range(episode_step, replay_buffer_size):
            child_conn.send(["actions", obs_zeros, agents_available_actions_zeros])
            child_conn.send(["replay_buffer", state_zeros, actions_zeros, [reward_zeros], [True], obs_zeros,
                             agents_available_actions_zeros, 1])
            child_conn.recv()

        child_conn.send(["episode_end", episode_reward, episode_step, env.win_counted])
    pass


class Runner:
    def __init__(self, arglist, scenario, actors):
        if arglist.algo_name == 'PCQMIX' or arglist.algo_name == 'PCVDN'or arglist.algo_name == 'PCQMIX-ICE':
            self.env = PCStarCraft2Env(map_name=scenario, replay_dir="./replay/")
        else:
            self.env = StarCraft2Env(map_name=scenario, replay_dir="./replay/")

        env_info = self.env.get_env_info()

        self.actors = actors
        self.scenario = scenario

        self.n_actions = env_info["n_actions"]  # 动作数
        self.n_agents = env_info["n_agents"]  # 智能体数
        self.state_shape = env_info["state_shape"]  # 状态空间大小
        self.obs_shape = env_info["obs_shape"] + self.n_agents + self.n_actions  # 观察空间大小
        self.episode_limit = env_info['episode_limit']  # 迭代限制,400
        self.unit_dim = env_info['unit_dim']
        self.episode_global_step = 0
        self.episode = 0

        if arglist.algo_name == 'IQL':
            self.algo = iql.IQL(arglist.train, self.n_agents, self.obs_shape, self.state_shape, self.n_actions,
                                0.0005, replay_buffer_size=1000)
        elif arglist.algo_name == 'VDN' or arglist.algo_name == 'PCVDN':
            self.algo = vdn.VDN(arglist.train, self.n_agents, self.obs_shape, self.state_shape, self.n_actions,
                                0.0005, replay_buffer_size=1000)
        elif arglist.algo_name == 'QMIX' or arglist.algo_name == 'PCQMIX':
            self.algo = qmix.QMix(arglist.train, self.n_agents, self.obs_shape, self.state_shape, self.n_actions,
                                  0.0005, replay_buffer_size=1000)
        elif arglist.algo_name == 'QMIX-ICE'or arglist.algo_name == 'PCQMIX-ICE':
            self.algo = qmix_ice.EQMix(arglist.train, self.n_agents, self.obs_shape, self.state_shape, self.n_actions,
                                  0.0005, replay_buffer_size=1000)

        if arglist.train == False:
            self.episode, self.episode_global_step = self.algo.load_model(
                '/home/xiong/github-submit/VF_ICE/models/' + arglist.algo_name + '/' + arglist.algo_name + '_' + arglist.scenario + arglist.r_F + '/agents_40000')
            print('Load model ' + arglist.algo_name + arglist.scenario + arglist.r_F + 'agents_40000')
            self.episode_global_step = 0
            self.episode = 0

        self.process_com = []
        self.locker = Lock()
        # actor表示进程数量，多进程创建过程
        for idx in range(self.actors):
            parent_conn, child_conn = Pipe()  # 进程间管道通信
            Process(target=env_run,
                    args=[arglist.algo_name, self.scenario, idx, child_conn, self.locker, self.episode_limit]).start()
            self.process_com.append(parent_conn)

        for process_conn in self.process_com:
            process_id = process_conn.recv()
            print(process_id, " is ready !")

        pass

    def reset(self):
        self.algo.on_reset(self.actors)
        self.episodes = []
        self.episode_reward = []
        self.episode_step = []
        self.replay_buffers = []
        self.win_counted_array = []
        episode_managed = self.episode
        for _ in range(self.actors):
            self.episodes.append(episode_managed)
            self.episode_reward.append(0)
            self.episode_step.append(0)
            self.win_counted_array.append(False)
            self.replay_buffers.append(qmix.ReplayBuffer(self.episode_limit))
            episode_managed += 1
        for process_conn in self.process_com:
            process_conn.send("Go !")

    def run(self):
        episode_done = 0
        process_size = len(self.process_com)
        available_to_send = np.array([True for _ in range(self.actors)])

        while True:
            obs_batch = []
            available_batch = []
            actions = None
            for idx, process_conn in enumerate(self.process_com):
                # if process_conn.poll():
                data = process_conn.recv()
                if data[0] == "actions":
                    obs_batch.append(data[1])
                    available_batch.append(data[2])

                    if idx == process_size - 1:
                        obs_batch = np.concatenate(obs_batch, axis=0)  # 数组按行拼接
                        available_batch = np.concatenate(available_batch, axis=0)
                        actions = self.algo.act(self.actors, torch.FloatTensor(obs_batch).to(device),
                                                torch.FloatTensor(available_batch).to(device))

                elif data[0] == "replay_buffer":
                    self.replay_buffers[idx].add(data[1], data[2], data[3], data[4], data[5], data[6],
                                                 data[7])  # 给第idx个进程的replaybuffer从头部压入数据

                elif data[0] == "episode_end":
                    self.episode_reward[idx] = data[1]
                    self.episode_step[idx] = data[2]
                    self.win_counted_array[idx] = data[3]
                    available_to_send[idx] = False
                    episode_done += 1

            if actions is not None:
                for idx_proc, process in enumerate(self.process_com):
                    if available_to_send[idx_proc]:
                        process.send(actions[idx_proc])

            if episode_done >= self.actors:
                break

        self.episode += self.actors  # 迭代集数以进程数为单位累加

        self.episode_global_step += max(self.episode_step)  # 一集最大的执行步数

        self.algo.decay_epsilon_greddy(self.episode_global_step)  # 得到epsilond的大小

        return self.replay_buffers

    def save(self):
        for process in self.process_com:
            process.send('save')
            data = process.recv()
            print(data)
        pass
        """
        for env in range(self.actors):
            env.save_replay()
        """

    def close(self):
        for process in self.process_com:
            process.send('close')
        pass
        """
        for env in range(self.actors):
            env.close()
        """
