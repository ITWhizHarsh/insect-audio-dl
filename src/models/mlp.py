"""Model A - multilayer perceptron on handcrafted acoustic features (classical baseline).

146-d summary statistics -> 512 -> 256 -> n_classes, with batch norm,
ReLU and dropout. The network has no access to time-frequency structure:
whatever it learns has to be visible in averaged MFCC / spectral statistics.
"""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(512, 256), dropout=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
