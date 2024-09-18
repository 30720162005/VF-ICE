import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class IQLer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim=64):
        super(IQLer, self).__init__()

    def update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)