"""Model C - convolutional-recurrent network (CNN front end + bidirectional LSTM).

Three conv blocks reduce the 128 x 431 log-mel image to a sequence of 53
frame vectors (128 channels x 16 mel bands each). A linear projection maps
each to 256-d, a 2-layer BiLSTM reads the sequence in both directions, and
attention pooling weights the time steps before classification.

Stridulation is a train of pulses with a species-specific repetition rate;
the CNN alone only sees that rhythm through a fixed receptive field, while
the LSTM can model it across the whole 5 s window. Parameter count (~1.6 M)
is kept close to the CNN so the comparison isolates the recurrent layer.

rnn="gru" swaps the LSTM for a GRU as an ablation. A GRU has three gates
instead of four, so the recurrent layers lose exactly a quarter of their
weights (~1.4 M parameters in total) and there is no separate cell state.
If it matches the LSTM, the long-term memory cell is not what the model
needs to follow the pulse rhythm over 5 s.

embed(x) returns the attention-pooled recurrent state (256-d) that the
classifier head reads; it is used for the embedding analysis.
"""
import torch
import torch.nn as nn

from src.models.cnn import conv_block

RNN = {"lstm": nn.LSTM, "gru": nn.GRU}


class CRNN(nn.Module):
    def __init__(self, n_classes, n_mels=128, channels=(32, 64, 128), hidden=128, dropout=0.3, rnn="lstm"):
        super().__init__()
        blocks, cin = [], 1
        for c in channels:
            blocks.append(conv_block(cin, c))
            cin = c
        self.features = nn.Sequential(*blocks)
        freq_out = n_mels // 2 ** len(channels)
        self.proj = nn.Sequential(nn.Linear(cin * freq_out, 256), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.rnn = RNN[rnn](256, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.att = nn.Linear(2 * hidden, 1)
        self.embed_dim = 2 * hidden
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.embed_dim, n_classes))

    def embed(self, x):                        # x: (B, 1, mels, frames)
        h = self.features(x)                   # (B, C, F', T')
        B, C, F, T = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B, T, C * F)
        h, _ = self.rnn(self.proj(h))          # (B, T', 2*hidden)
        w = torch.softmax(self.att(h), dim=1)  # attention weights over time
        return (w * h).sum(dim=1)

    def forward(self, x):
        return self.fc(self.embed(x))
