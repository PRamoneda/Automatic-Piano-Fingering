import pdb

import torch
from torch import nn

from nns.common import without_embedding
import torch.nn.functional as F


class nakamura_encoder(nn.Module):
    # the encoder / feature extractor / generator? xd
    def __init__(self, input):
        super(nakamura_encoder, self).__init__()
        self.FF1 = nn.Linear(input, 32)
        self.FF2 = nn.Linear(32, 32)
        self.FF3 = nn.Linear(32, 5)

    def forward(self, x, x_lengths=None, edge_list=None):
        # pdb.set_trace()
        x = self.FF1(x.float())
        x = self.FF2(x.float())
        x = self.FF3(x.float())
        return x


class nakamura_encoderLSTM(nn.Module):
    # the encoder / feature extractor / generator? xd
    def __init__(self, input):
        super(nakamura_encoderLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input, hidden_size=32, num_layers=3)
        self.FF = nn.Linear(32, 5)

    def forward(self, x, x_lengths=None, edge_list=None):
        # pdb.set_trace()
        x, _ = self.rnn(x.float())
        x = self.FF(x.float())
        return x



class AR_GRU_decoder(nn.Module):
    def __init__(self, in_size):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super(AR_GRU_decoder, self).__init__()
        # parameters
        self.in_size = in_size
        self.hidden_size = 32
        # nn
        self.ar_gru = nn.GRU(
            input_size=self.in_size * 2,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.FC = nn.Linear(32, 5)
        self.class_embedding = nn.Embedding(5, in_size)

    def forward(self, x, x_lengths, gt_label=False):
        if not isinstance(gt_label, bool):
            # for training (teacher forcing)
            gt_label = torch.unsqueeze(gt_label, dim=2)
            prev_gt = torch.cat(
                (torch.zeros((gt_label.shape[0], 1, gt_label.shape[2]), device=self.device, dtype=torch.long),
                 gt_label[:, :-1, :].type(torch.LongTensor).to(self.device)),
                dim=1)
            embed_prev = self.class_embedding(prev_gt)

            concated_data = torch.cat((x, torch.squeeze(embed_prev, dim=2)), dim=2)
            # pdb.set_trace()
            result, _ = self.ar_gru(concated_data)
            result = self.FC(result)
            total_result = F.log_softmax(result, dim=2)
        else:
            # for testing (to instantiate)
            h = self.init_hidden(x.shape[0])
            prev_out = torch.zeros((x.shape[0], 1, self.in_size)).to(self.device)
            total_result = torch.zeros((x.shape[0], x.shape[1], 5)).to(self.device)
            for i in range(x.shape[1]):
                out = torch.cat((x[:, i:i + 1, :], prev_out), dim=2)
                out, h = self.ar_gru(out, h)
                out = self.FC(out)
                current_out = F.log_softmax(out, dim=2)
                prev_out = torch.argmax(current_out, dim=2)
                prev_out = self.class_embedding(prev_out)
                total_result[:, i:i + 1, :] = current_out
        return total_result


class windowed(nn.Module):
    def __init__(self, embedding=without_embedding(), encoder=nakamura_encoder(3), decoder=nn.Linear(32, 10)):
        super(windowed, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
        # next forward parameter
        x = self.embedding(notes, onsets, durations, x_lengths)
        x = self.encoder(x, x_lengths, edge_list)
        if fingers is not None:
            logits = self.decoder(x, x_lengths, fingers)
        else:
            logits = self.decoder(x, x_lengths)

        return logits


