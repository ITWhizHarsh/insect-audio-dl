"""Model B - VGG-style CNN on log-mel spectrograms.

Four blocks of [conv3x3-BN-ReLU] x 2 + 2x2 average pooling (32-64-128-256
channels). After the last block the frequency axis is averaged out and the
time axis is summarised by mean + max pooling, so the classifier sees both
how strong a pattern is on average and its single strongest occurrence
(useful for intermittent calls). About 1.2 M parameters.

The EfficientNetV2-S transfer-learning variant is added in the final phase.
"""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.AvgPool2d(2),
    )


class LogMelCNN(nn.Module):
    def __init__(self, n_classes, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        blocks, cin = [], 1
        for c in channels:
            blocks.append(conv_block(cin, c))
            cin = c
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * cin, n_classes)

    def forward(self, x):                      # x: (B, 1, mels, frames)
        h = self.features(x).mean(dim=2)       # (B, C, T')
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.fc(self.dropout(h))
