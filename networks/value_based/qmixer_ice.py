import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_cuda = th.cuda.is_available()
device = th.device("cuda" if use_cuda else "cpu")

class EQMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim=64):
        super(EQMixer, self).__init__()

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
        bs = agent_qs.size(0)#batch size:16
        states = states.reshape(-1, self.state_dim)#(16*episode_len) * state_dim
        agent_qs = agent_qs.view(-1, 1, self.n_agents)#(16*episode_len) * 1 * 智能体数量

        # 计算权重
        w_c = self.wc_calculate(agent_qs, states, bs)
        # 根据权重得到新的智能体的个体q
        agent_qs = th.mul(agent_qs, w_c)

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

    def wc_calculate(self,agent_qs,states,bs):
        ##计算初始q_tot
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)  # (16*episode_len) * 6 * 32
        b1 = b1.view(-1, 1, self.embed_dim)  # (16*epis ode_len) * 1 * 32
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)  # th矩阵乘法；elu激活函数
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)  # (16*episode_len) * 32 * 1
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)  # (16*episode_len) * 1 * 1
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        ####计算对应的剔除q_1_tot，并归一化差值作为权重
        dim0 = states.size(0)
        dim1 = self.n_agents
        dim2 = int(self.state_dim / self.n_agents)
        #state_0 = states.reshape(dim0, dim1, dim2)
        #state_1=th.zeros((dim0, dim1, dim2),dtype=th.float).to(device)
        state_1 = th.zeros((dim0, self.state_dim), dtype=th.float).to(device)
        wc = th.zeros((bs,q_tot.size(1),self.n_agents),dtype=th.float).to(device)
        for i in range(self.n_agents):
            state_1[:,:] = states[:,:].detach()
            state_1=state_1.reshape(dim0, dim1, dim2)
            state_1[:, i, :] = 0
            state_1 = state_1.reshape(-1, self.state_dim)
            w_1 = th.abs(self.hyper_w_1(state_1))
            b_1 = self.hyper_b_1(state_1)
            w_1 = w_1.view(-1, self.n_agents, self.embed_dim)  # (16*episode_len) * 6 * 32
            b_1 = b_1.view(-1, 1, self.embed_dim)  # (16*epis ode_len) * 1 * 32
            hidden_1 = F.elu(th.bmm(agent_qs, w_1) + b_1)  # th矩阵乘法；elu激活函数
            # Second layer
            w_1_final = th.abs(self.hyper_w_final(state_1))
            w_1_final = w_1_final.view(-1, self.embed_dim, 1)  # (16*episode_len) * 32 * 1
            # State-dependent bias
            v_1 = self.V(state_1).view(-1, 1, 1)  # (16*episode_len) * 1 * 1
            # Compute final output
            y_1 = th.bmm(hidden_1, w_1_final) + v_1
            # Reshape and return
            q_1_tot = y_1.view(bs, -1, 1)
            wc[:,:,i] = th.abs(q_tot - q_1_tot).reshape(bs, -1)
        wc = th.nn.functional.normalize(wc, dim=2)
        wc=wc.reshape(-1,1,self.n_agents).detach()
        return wc

    def update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)