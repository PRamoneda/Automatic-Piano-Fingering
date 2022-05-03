# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import pdb

import torch
from torch import nn
from torch.nn import functional as F


def build_model(model_name, input_lengths, hidden_size, num_class):
    if model_name == 'LSTM':
        model = LSTMModel(input_size=input_lengths, input_dim=1, num_layers=3, hidden_size=hidden_size, output_size=num_class)
    elif model_name == 'Forward':
        model = ForwardNNModel(input_size=input_lengths, input_dim=1, num_layers=3, hidden_size=hidden_size, output_size=num_class)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.apply(init_weight)
    return model


def init_weight(m):
    """
    To initialize weights and biases.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)

    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)


class LSTMModel(nn.Module):
    def __init__(self, input_size, input_dim, hidden_size, output_size, num_layers=1, bidirectional=False):
        """
        Args:
            input_size: (unsign int). The lengths of x.
            input_dim: The dimension of x
            output_size: (unsign int). The number of predicted classes
            hidden_size: (unsign int). The number of cells of LSTM layer.
            num_layers: (unsign int). The number of layers of LSTM layer.
            bidrectional: bool. Bidretional LSTM or unidirectional LSTM.
        """
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        num_directions = 2 if bidirectional else 1
        in_features = input_size * num_directions * hidden_size
        self.output_linear = nn.Linear(in_features=in_features, out_features=output_size)

    def forward(self, notes, onsets, durations, lengths, edge_list, log_soft_max=True):
        batch_size = notes.size(0)
        notes = notes
        # pdb.set_trace()
        lstm_out, (_, _) = self.lstm(notes)
        lstm_out = lstm_out.contiguous().view(batch_size, -1)

        logit_out = self.output_linear(lstm_out)
        if log_soft_max:
            return F.log_softmax(logit_out, dim=1)
        else:
            return logit_out


class ForwardNNModel(nn.Module):
    def __init__(self, input_size, input_dim, hidden_size, output_size, num_layers=1):
        """
        Args:
            input_size: (unsign int). The lengths of x.
            input_dim: The dimension of x
            output_size: (unsign int). The number of predicted classes
            hidden_size: (unsign int). The number of cells of linear layers expect for output_linear.
            num_layers: (unsign int). The number of layers of linear layers.
        """
        super(ForwardNNModel, self).__init__()
        linear_list = []
        for i in range(num_layers):
            in_size = (input_size * input_dim) if i == 0 else hidden_size
            l = nn.Linear(in_size, hidden_size)
            a = nn.Sigmoid()#nn.ReLU()
            linear_list.extend([l, a])
            # linear_list.extend([l])

        self.linears = nn.ModuleList(linear_list)
        self.output_linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, onsets, durations, lengths, edge_list):
        x = torch.squeeze(x, dim=2)
        for i, l in enumerate(self.linears):
            x = l(x)
        # pdb.set_trace()
        logit_out = self.output_linear(x)

        return F.log_softmax(logit_out, dim=1)
