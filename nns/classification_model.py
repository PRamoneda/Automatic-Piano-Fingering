import pdb

import torch
from torch import nn

from nns import GGCN, common
from nns.GGCN import GatedGraph
from nns.common import without_embedding
import torch.nn.functional as F
from torchlayers.regularization import L1, L2


def l1l2(x):
    return L2(L1(x, weight_decay=1e-5), weight_decay=1e-4)


class embeddingVicent(nn.Module):
    # simple embedding inspired in the two violin-fingering papers of 2021
    # https://github.com/vkmcheung/violin-ssvae/blob/63bb104ec2363f831b36e448ebc9577b04b462ea/main.py#L205
    #  in the continous variables we have changed
    # TODO We have to personalize this embedding layer in the future
    out_dim = 64

    def __init__(self):
        super(embeddingVicent, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.note_embedding = l1l2(nn.Embedding(91, 16))
        self.onset_embedding = l1l2(nn.Linear(1, 8))
        self.duration_embedding = l1l2(nn.Linear(1, 4))
        self.dense = l1l2(nn.Linear(in_features=28, out_features=64))
        self.PReLU = nn.PReLU()
        self.LayerNorm = nn.LayerNorm(normalized_shape=64)

    def forward(self, notes, onsets, durations, x_lengths):
        note_emb = torch.squeeze(self.note_embedding(notes.int()), dim=2)
        onset_emb = self.onset_embedding(onsets)
        duration_emb = self.duration_embedding(durations)
        # pdb.set_trace()
        x = torch.cat([note_emb, onset_emb, duration_emb], dim=2)
        x = self.dense(x)
        x = self.PReLU(x)
        x = self.LayerNorm(x)
        return x


class embeddingSimple(nn.Module):
    # simple embedding only adding a 4 embedding for the note
    out_dim = 5

    def __init__(self):
        super(embeddingSimple, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # the fault
        self.note_embedding = l1l2(nn.Embedding(91, 8))

    def forward(self, notes, onsets, durations, x_lengths):
        embedding = torch.squeeze(self.note_embedding(notes.int()), dim=2)
        note_emb = torch.concat([embedding, onsets, durations], dim=2)
        return note_emb


class fc_encoder(nn.Module):
    # the encoder / feature extractor / generator? xd
    def __init__(self, input, output=5, units=32, windows_len=11):
        super(fc_encoder, self).__init__()

        layers = [
            nn.Linear(input * windows_len, units),
            nn.ReLU(inplace=True),
            nn.Linear(units, units),
            nn.ReLU(inplace=True),
            nn.Linear(units, output),
            # nn.ReLU(inplace=True)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, x_lengths=None, edge_list=None):
        x = x.view(x.shape[0], -1)
        x = self.layers(x.float())
        return F.log_softmax(x, dim=1)


class gnn_encoder(nn.Module):
    def __init__(self, input_size, units=32):
        super(gnn_encoder, self).__init__()
        self.FC = nn.Linear(input_size, units)
        self.gnn1 = GatedGraph(size=units, secondary_size=units, num_edge_types=GGCN.N_EDGE_TYPE)
        self.gnn2 = GatedGraph(size=units, secondary_size=units, num_edge_types=GGCN.N_EDGE_TYPE)
        self.gnn3 = GatedGraph(size=units, secondary_size=units, num_edge_types=GGCN.N_EDGE_TYPE)
        self.out = nn.Linear(32, 5)

    def forward(self, x_padded, x_lengths, edges):
        x = self.FC(x_padded)
        x = self.gnn1(x, edges)
        x = self.gnn2(x, edges)
        x = self.gnn3(x, edges)
        x = x[:, 5]
        x = self.out(x)
        return F.log_softmax(x, dim=1)

    def freeze_l0(self):
        for param in self.FC.parameters():
            param.requires_grad = False
        for param in self.gnn1.parameters():
            param.requires_grad = False

    def freeze_l0l1(self):
        for param in self.FC.parameters():
            param.requires_grad = False
        for param in self.gnn1.parameters():
            param.requires_grad = False
        for param in self.gnn2.parameters():
            param.requires_grad = False


class lstm_encoder(nn.Module):
    # the encoder / feature extractor / generator? xd
    def __init__(self, input, dropout=0.0):
        super(lstm_encoder, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input, hidden_size=32, batch_first=True, num_layers=3, bidirectional=True,
                            dropout=dropout)
        self.FC = nn.Linear(64, 5)

    def forward(self, x, x_lengths=None, edge_list=None):
        # pdb.set_trace()
        x, _ = self.rnn1(x.float())
        x = x[:, 5]
        x = self.FC(x)
        # only take the reverse time direction output (which have the time direction as input)
        pdb.set_trace()
        return F.log_softmax(x, dim=1)

    def freeze_l0(self):
        relevant_parameters = [i for i, (param_name, param_value) in
                               enumerate(list(self.rnn1.named_parameters()))
                               if 'l0' in param_name]

        for i, cur_parameter in enumerate(self.rnn1.parameters()):
            if i in relevant_parameters:
                print("Setting for {0}".format(i))
                cur_parameter.requires_grad = False

    def freeze_l0l1(self):
        relevant_parameters = [i for i, (param_name, param_value) in
                               enumerate(list(self.rnn1.named_parameters()))
                               if 'l0' in param_name or 'l1' in param_name]

        for i, cur_parameter in enumerate(self.rnn1.parameters()):
            if i not in relevant_parameters:
                print("Setting for {0}".format(i))
                cur_parameter.requires_grad = False


class classification(nn.Module):
    def __init__(self, embedding=without_embedding(), encoder=fc_encoder(3), decoder=nn.Linear(32, 10)):
        super(classification, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
        # next forward parameter
        # pdb.set_trace()
        x = self.embedding(notes, onsets, durations, x_lengths)
        x = self.encoder(x, x_lengths, edge_list)
        # pdb.set_trace()
        return x

    def freeze_all(self):
        for param in self.embedding.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_l0(self):
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.encoder.freeze_l0()

    def freeze_l0l1(self):
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.encoder.freeze_l0l1()

    def unfreeze_last_layer(self):
        self.encoder.unfreeze_last_layer()


if __name__ == '__main__':
    model = classification(
        embedding=common.only_pitch(),
        encoder=lstm_encoder(input=1, dropout=.4),
    )
    model.freeze_all()
    model.unfreeze_last_layer()
    for p in model.parameters():
        print(p.requires_grad)

