import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.first = nn.Conv1d(width, width, 3, dilation=dilation)
        self.second = nn.Conv1d(width, width, 3, dilation=dilation)

    def forward(self, x):
        h = F.relu(self.first(F.pad(x, (2 * self.dilation, 0))))
        return F.relu(x + self.second(F.pad(h, (2 * self.dilation, 0))))


class Encoder(nn.Module):
    def __init__(self, features, width=64):
        super().__init__()
        self.input = nn.Conv1d(features, width, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8)])

    def forward(self, x):
        return self.blocks(self.input(x.transpose(1, 2)))[:, :, -1]


class BuildingModel(nn.Module):
    def __init__(self, features, width=64):
        super().__init__()
        self.encoder = Encoder(features, width)
        self.head = nn.Linear(width, 4)

    def forward(self, x):
        raw = self.head(self.encoder(x))
        median = raw[:, 0]
        return median, median - F.softplus(raw[:, 1]), median + F.softplus(raw[:, 2]), raw[:, 3]


def objective(prediction, y, anomaly):
    median, low, high, logit = prediction
    residual_low, residual_high = y - low, y - high
    pinball = torch.maximum(0.05 * residual_low, -0.95 * residual_low).mean()
    pinball = pinball + torch.maximum(0.95 * residual_high, -0.05 * residual_high).mean()
    return F.huber_loss(median, y) + 0.5 * pinball + 0.2 * F.binary_cross_entropy_with_logits(logit, anomaly)
