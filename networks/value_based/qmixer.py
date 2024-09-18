import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim=64):
        super(QMixer, self).__init__()

        #self.args = args
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

        self.embed_dim = mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents) #线性层 282->32*6
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim) #线性层 282->32

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim) #线性层 282->32

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)#16
        states = states.reshape(-1, self.state_dim)#(16*episode_len) * 282
        agent_qs = agent_qs.view(-1, 1, self.n_agents)#(16*episode_len) * 1 * 6
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim) #(16*episode_len) * 6 * 32
        b1 = b1.view(-1, 1, self.embed_dim) #(16*epis ode_len) * 1 * 32
        hidden = F.elu(th.bmm(agent_qs, w1) + b1) #th矩阵乘法；elu激活函数
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)#(16*episode_len) * 32 * 1
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1) #(16*episode_len) * 1 * 1
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

    def update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)