import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as F


class only_pitch(nn.Module):
    out_dim = 1
    def __init__(self):
        super(only_pitch, self).__init__()

    def forward(self, notes, onsets, durations, x_lengths):
        return notes


class emb_pitch(nn.Module):
    out_dim = 64

    def __init__(self):
        super(emb_pitch, self).__init__()
        self.emb = nn.Embedding(127, 64)

    def forward(self, notes, onsets, durations, x_lengths):
        # pdb.set_trace()
        return torch.squeeze(self.emb((notes * 127).int()), dim=2)


class without_embedding(nn.Module):
    out_dim = 3
    def __init__(self):
        super(without_embedding, self).__init__()

    def forward(self, notes, onsets, durations, x_lengths):
        return torch.concat([notes, onsets, durations], dim=2)


class without_decoder(nn.Module):
    def __init__(self):
        super(without_decoder, self).__init__()

    def forward(self, x, x_lengths):
        # pdb.set_trace()
        return F.log_softmax(x, dim=2)


class linear_decoder(nn.Module):
    def __init__(self, in_size):
        super(linear_decoder, self).__init__()
        self.FC = nn.Linear(in_size, 5)

    def forward(self, x, x_lengths):
        x = self.FC(x)
        return F.log_softmax(x, dim=2)


