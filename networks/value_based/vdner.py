import torch as th
import torch.nn as nn

import torch.nn.functional as F
import numpy as np


class VDNMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim=64):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs):
        return th.sum(agent_qs, dim=2, keepdim=True)

    def update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)