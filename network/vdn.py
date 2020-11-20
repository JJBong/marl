import torch.nn as nn
import torch


class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet, self).__init__()

    @staticmethod
    def forward(q_values):
        return torch.sum(q_values, dim=1)
