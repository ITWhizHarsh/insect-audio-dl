"""Model B2 - EfficientNetV2-S on log-mel spectrograms, transfer-learned from ImageNet.

This is the heavy member of the CNN family. The VGG-style CNN in cnn.py has
about 1.2 M parameters and starts from random weights; EfficientNetV2-S
(Tan & Le, ICML 2021) has about 20 M and starts from ImageNet. Comparing the
two asks whether a much deeper, pretrained CNN is worth it when many species
have only a handful of training recordings.

A spectrogram is not a photograph, but the first layers of an ImageNet
network learn generic detectors for edges, blobs and repeated textures, and
in a log-mel image those are call onsets, harmonic lines and pulse trains.
Starting from such filters and fine-tuning everything is the transfer recipe
that PANNs (Kong et al., 2020) use from AudioSet and PSLA (Gong et al., 2021)
use from ImageNet with EfficientNet; it converges faster and overfits less
than training a large network from scratch on a small labelled set.

ImageNet expects three colour channels. Instead of repeating the spectrogram
three times, the 3-channel stem convolution is replaced by a 1-channel one
whose kernel is the pretrained RGB kernel summed over its input channels.
Convolution is linear, so this gives exactly the same output as the repeated
input while the stem does a third of the work. The inputs are standardised
with the training mean/std, which puts them on roughly the same scale as
ImageNet-normalised images.

The 128 x 431 input leaves a 4 x 14 map of 1280 channels after the backbone;
global average pooling gives the 1280-d embedding, followed by dropout and a
new linear layer. Stochastic depth (randomly skipping residual blocks during
training) is kept from torchvision as a regulariser and can be set here.

Per chunk it does about as much arithmetic as the small CNN (roughly 6.3
against 5.6 GFLOPs for one forward pass), but it is 40 blocks (some 170
layers) deep and widens the channels four to six times inside most blocks, so
training has to keep far more activations in memory and is limited by memory
traffic rather than arithmetic. On the 8 GB M2 it is therefore slow to
fine-tune, and the final runs may be done on a cloud GPU.
"""
import torch
import torch.nn as nn


def mono_stem(conv):
    """1-channel copy of the RGB stem: conv(x repeated 3 times, W) == conv(x, W summed over RGB)."""
    mono = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    with torch.no_grad():
        mono.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
    return mono


class EfficientNetV2S(nn.Module):
    def __init__(self, n_classes, pretrained=True, dropout=0.3, stochastic_depth=0.2):
        super().__init__()
        # imported here so that importing src.models never depends on torchvision
        from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_v2_s(weights=weights, stochastic_depth_prob=stochastic_depth)
        net.features[0][0] = mono_stem(net.features[0][0])
        self.features = net.features           # the ImageNet classifier is dropped
        self.embed_dim = net.classifier[1].in_features   # 1280
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.embed_dim, n_classes)

    def embed(self, x):                        # x: (B, 1, mels, frames)
        return self.features(x).mean(dim=(2, 3))   # global average pool -> (B, 1280)

    def forward(self, x):
        return self.fc(self.dropout(self.embed(x)))
