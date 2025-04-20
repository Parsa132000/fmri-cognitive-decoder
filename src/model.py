import torch
import torch.nn as nn


class BrainNet(nn.Module):
    """
    A simple fully-connected neural network for fMRI classification.
    """

    def __init__(self, input_dim, num_classes):
        super(BrainNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
# Placeholder for model.py
