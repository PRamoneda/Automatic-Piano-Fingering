import pdb
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder

import common_classification
from common_classification import load_model, save_model
from finetuning_seq2seq import create_dataset
from loader import salami_tensor
from nns import GGCN, classification_model, common
import torch.nn.functional as F





# class lstm_transfer_base():


class lstm_transfer_equal(nn.Module):
    def __init__(self,  freeze_all_=True):
        super(lstm_transfer_equal, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.pretrained_model = classification_model.classification(
            embedding=common.only_pitch(),
            encoder=classification_model.lstm_encoder(input=1, dropout=.4),
        )
        self.pretrained_model, _, _, _ = load_model(
            path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:lstm(dropout2)).pth',
            model=self.pretrained_model,
            device=self.device
        )
        if freeze_all_:
            self.pretrained_model.freeze_all()

    def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
        # pretrained get only pitch (without parameters)
        x_padded = self.pretrained_model.embedding(notes, onsets, durations, x_lengths)
        # to handle different lengths
        x_packed = packer(x_padded.float(), x_lengths.cpu().numpy(), batch_first=True)
        # the pretrained lstm
        output, _ = self.pretrained_model.encoder.rnn1(x_packed)
        # to handle different lengths
        output_padded, _ = padder(output, batch_first=True)
        x = self.pretrained_model.encoder.FC(output_padded)
        # pdb.set_trace()
        return F.log_softmax(x, dim=2)



# class lstm_transfer(nn.Module):
#
#     def __init__(self, decoder=lstm_decoder, dropout=0):
#         super(lstm_transfer, self).__init__()
#
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
#         model = classification_model.classification(
#             embedding=common.only_pitch(),
#             encoder=classification_model.lstm_encoder(input=1, dropout=.4),
#         )
#         self.pretrained_model, _, _, _ = load_model(
#             path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:lstm(dropout2)).pth',
#             model=model,
#             device=self.device
#         )
#         self.pretrained_model.freeze_all()
#         self.decoder = decoder(dropout)
#
#     def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
#         # pretrained get only pitch (without parameters)
#         x_padded = self.pretrained_model.embedding(notes, onsets, durations, x_lengths)
#         # to handle different lengths
#         x_packed = packer(x_padded.float(), x_lengths.cpu().numpy(), batch_first=True)
#         # the pretrained lstm
#         output, _ = self.pretrained_model.encoder.rnn1(x_packed)
#         # to handle different lengths
#         output_padded, _ = padder(output, batch_first=True)
#         x = self.decoder(output_padded, x_lengths, None)
#         # pdb.set_trace()
#         return x


# class lstm_transfer_l0l1(nn.Module):
#
#     def __init__(self, decoder=lstm_decoder, dropout=0):
#         super(lstm_transfer_l0l1, self).__init__()
#
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
#         model = classification_model.classification(
#             embedding=common.only_pitch(),
#             encoder=classification_model.lstm_encoder(input=1, dropout=.4),
#         )
#         self.pretrained_model, _, _, _ = load_model(
#             path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:lstm(dropout2)).pth',
#             model=model,
#             device=self.device
#         )
#         relevant_parameters = [i for i, (param_name, param_value) in
#                                enumerate(list(self.pretrained_model.encoder.rnn1.named_parameters()))
#                                if 'l2' in param_name]
#         for i, cur_parameter in enumerate(self.pretrained_model.encoder.rnn1.parameters()):
#             if i in relevant_parameters:
#                 print("Setting for {0}".format(i))
#                 cur_parameter.requires_grad = False
#         # self.decoder = decoder(dropout)
#         self.out = nn.Linear(64, 5)
#
#     def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
#         # pretrained get only pitch (without parameters)
#         x_padded = self.pretrained_model.embedding(notes, onsets, durations, x_lengths)
#         # to handle different lengths
#         x_packed = packer(x_padded.float(), x_lengths.cpu().numpy(), batch_first=True)
#         # the pretrained lstm
#         output, _ = self.pretrained_model.encoder.rnn1(x_packed)
#         # to handle different lengths
#         output_padded, _ = padder(output, batch_first=True)
#         output_padded = self.out(output_padded)
#         return F.log_softmax(output_padded, dim=2)
#
#
#
# class lstm_transfer_gnn(nn.Module):
#     def __init__(self, decoder=lstm_decoder, dropout=0):
#         super(lstm_transfer_gnn, self).__init__()
#
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
#         model = classification_model.classification(
#             embedding=common.only_pitch(),
#             encoder=classification_model.lstm_encoder(input=1, dropout=.4),
#         )
#         self.pretrained_model, _, _, _ = load_model(
#             path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:lstm(dropout2)).pth',
#             model=model,
#             device=self.device
#         )
#         self.gnn = gnn_decoder(size=64)
#         # self.decoder = decoder(dropout)
#         self.out = nn.Linear(64, 5)
#
#     def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
#         # pretrained get only pitch (without parameters)
#         x_padded = self.pretrained_model.embedding(notes, onsets, durations, x_lengths)
#         # to handle different lengths
#         x_packed = packer(x_padded.float(), x_lengths.cpu().numpy(), batch_first=True)
#         # the pretrained lstm
#         output, _ = self.pretrained_model.encoder.rnn1(x_packed)
#         # to handle different lengths
#         output_padded, _ = padder(output, batch_first=True)
#         output_padded = self.gnn(output_padded, x_lengths, edge_list)
#         output_padded = self.out(output_padded)
#         return F.log_softmax(output_padded, dim=2)




# class lstm_transfer_AR(nn.Module):
#     def __init__(self):
#         super(lstm_transfer_AR, self).__init__()
#
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
#         model = classification_model.classification(
#             embedding=common.only_pitch(),
#             encoder=classification_model.lstm_encoder(input=1, dropout=.4),
#         )
#         self.pretrained_model, _, _, _ = load_model(
#             path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:lstm(dropout2)).pth',
#             model=model,
#             device=self.device
#         )
#         self.decoder = AR_decoder(in_size=64, ratio_teaching_forcing=0.4)
#         # self.decoder = decoder(dropout)
#
#     def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
#         # pretrained get only pitch (without parameters)
#         x_padded = self.pretrained_model.embedding(notes, onsets, durations, x_lengths)
#         # to handle different lengths
#         x_packed = packer(x_padded.float(), x_lengths.cpu().numpy(), batch_first=True)
#         # the pretrained lstm
#         output, _ = self.pretrained_model.encoder.rnn1(x_packed)
#         # to handle different lengths
#         output_padded, _ = padder(output, batch_first=True)
#         output_padded = self.decoder(output_padded, x_lengths, fingers)
#         return output_padded



# if __name__ == '__main__':
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     _, _, test_rh, test_lh, windowed = create_dataset('augmented_seq2seq')
#     model = lstm_transfer_equal()
#
#     model.to(device)
#     acc_rh = common_classification.compute_results_seq2seq(None, test_rh, model, device, None, False)
#     acc_lh = common_classification.compute_results_seq2seq(None, test_lh, model, device, None, False)
#     print(f"Test (General match rate): rh:{acc_rh:2.2%} lh:{acc_lh:2.2%}")
#
#     # rnn_trained = nn.encoder(input_size=1, hidden_size=32, num_layers=3, batch_first=True, bidirectional=True)
#     #
#     # rnn_layers = [
#     #     nn.encoder(input_size=1, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True),
#     #     nn.encoder(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True),
#     #     nn.encoder(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
#     # ]


########################################################################
########################################################################
def get_pretrained(arq, soft, device):
    if arq == "LSTM" and soft:
        model = classification_model.classification(
            embedding=common.only_pitch(),
            encoder=classification_model.lstm_encoder(input=1, dropout=.4),
        )
        pretrained_model, _, _, _ = load_model(
            path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:lstm(dropout2)).pth',
            model=model,
            device=device
        )
    elif arq == "LSTM" and not soft:
        model = classification_model.classification(
            embedding=common.only_pitch(),
            encoder=classification_model.lstm_encoder(input=1),
        )
        pretrained_model, _, _, _ = load_model(
            path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#only_pitch:lstm.pth',
            model=model,
            device=device
        )
    elif arq == "GNN":
        model = classification_model.classification(
            embedding=common.only_pitch(),
            encoder=classification_model.gnn_encoder(input_size=1),
        )
        if soft:
            pretrained_model, _, _, _ = load_model(
                path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:gnn).pth',
                model=model,
                device=device
            )
        else:
            pretrained_model, _, _, _ = load_model(
                path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#only_pitch:gnn.pth',
                model=model,
                device=device
            )
    return pretrained_model

class encoder_pretrained(nn.Module):
    def __init__(self, arq, freeze_layers, soft=True):
        super(encoder_pretrained, self).__init__()
        self.arq = arq

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        pretrained_model = get_pretrained(arq, soft, self.device)

        if freeze_layers == "3":
            pretrained_model.freeze_all()
        elif freeze_layers == "2":
            pretrained_model.freeze_l0l1()
        elif freeze_layers == "1":
            pretrained_model.freeze_l0()
        elif freeze_layers == "0":
            pass
        else:
            raise "freeze layers strange"

        self.embedding = pretrained_model.embedding
        self.encoder = pretrained_model.encoder


    def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
        # pretrained get only pitch (without parameters)
        x_padded = self.embedding(notes, onsets, durations, x_lengths)
        # to handle different lengths

        if self.arq == 'LSTM':
            x_packed = packer(x_padded.float(), x_lengths.cpu().numpy(), batch_first=True)
            # the pretrained lstm
            output, _ = self.encoder.rnn1(x_packed)
            # to handle different lengths
            output, _ = padder(output, batch_first=True)
        elif self.arq == 'GNN':
            x = self.encoder.FC(x_padded.float())
            x = self.encoder.gnn1(x, edge_list)
            x = self.encoder.gnn2(x, edge_list)
            output = self.encoder.gnn3(x, edge_list)
        # pdb.set_trace()
        return output


class FC_pretrained(nn.Module):

    def __init__(self, encoder_type):
        super(FC_pretrained, self).__init__()
        arq = "GNN" if "GNN" in encoder_type else "LSTM"
        soft = "soft" in encoder_type


        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        pretrained_model = get_pretrained(arq, soft, self.device)
        if arq == 'LSTM':
            self.FC = pretrained_model.encoder.FC
        elif arq == "GNN":
            self.FC = pretrained_model.encoder.out


    def forward(self, x, x_lengths, edge_list, fingers):
        x = self.FC(x)
        return F.log_softmax(x, dim=2)


class lstm_decoder(nn.Module):
    # the encoder / feature extractor / generator? xd
    def __init__(self, input_size, dropout=0):
        super(lstm_decoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(64, 5)

    def forward(self, x_padded, x_lengths, edge_list, fingers):
        x_packed = packer(x_padded.float(), x_lengths.cpu().numpy(), batch_first=True)
        output, _ = self.lstm1(x_packed)
        output_padded, _ = padder(output, batch_first=True)
        output_padded = self.out(output_padded)
        return F.log_softmax(output_padded, dim=2)


class gnn_decoder(nn.Module):
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gated_graph_conv.html#GatedGraphConv
    # https://arxiv.org/pdf/1511.05493.pdf
    def __init__(self, input_size):
        super(gnn_decoder, self).__init__()
        self.gnn1 = GGCN.GatedGraph(size=input_size, secondary_size=input_size, num_edge_types=GGCN.N_EDGE_TYPE)
        self.out = nn.Linear(input_size, 5)

    def forward(self, x_padded, x_lengths, edges, fingers):
        # x_packed = packer(x_padded, x_lengths, batch_first=True)
        output = self.gnn1(x_padded, edges)
        # output_padded, _ = padder(output, batch_first=True)
        output = self.out(output)
        return F.log_softmax(output, dim=2)


class AR_decoder(nn.Module):
    def __init__(self, in_size, ratio_teaching_forcing):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super(AR_decoder, self).__init__()
        # parameters
        self.in_size = in_size
        self.hidden_size = 32
        self.num_layers = 1
        self.ratio_teaching_forcing = ratio_teaching_forcing
        # nn
        self.ar_lstm = nn.LSTM(
            input_size=self.in_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.FC = nn.Linear(32, 5)
        self.class_embedding = nn.Embedding(5, in_size)

    def forward(self, x, x_lengths, edge_list, gt_label=None):
        teaching_forcing = False
        # pdb.set_trace()
        if gt_label is not None:
            teaching_forcing = random.random() < self.ratio_teaching_forcing

        if teaching_forcing:
            # print("teaching forcing")
            gt_label = torch.unsqueeze(gt_label, dim=2)
            prev_gt = torch.cat(
                (torch.zeros((gt_label.shape[0], 1, gt_label.shape[2]), device=self.device, dtype=torch.long),
                 gt_label[:, :-1, :].type(torch.LongTensor).to(self.device)),
                dim=1)
            prev_gt_packed = packer(prev_gt, x_lengths.cpu().numpy(), batch_first=True)
            embed_previous_packed = self.elementwise(self.class_embedding, prev_gt_packed)
            embed_previous_padded, _ = padder(embed_previous_packed, batch_first=True)
            concated_data = torch.cat((x, torch.squeeze(embed_previous_padded, dim=2)), dim=2)
            concated_data_packed = packer(concated_data.float(), x_lengths.cpu().numpy(), batch_first=True)
            result, _ = self.ar_lstm(concated_data_packed)
            result_padded, _ = padder(result, batch_first=True)
            result_padded = self.FC(result_padded)
            # pdb.set_trace()
            total_result = F.log_softmax(result_padded, dim=2)
        else:
            # print("no teaching forcing")
            # for testing (to instantiate)
            (hh, cc) = self.init_hidden(x.shape[0])
            prev_out = torch.zeros((x.shape[0], 1, self.in_size)).to(self.device)
            total_result = torch.zeros((x.shape[0], x.shape[1], 5)).to(self.device)
            for i in range(x.shape[1]):
                out = torch.cat((x[:, i:i + 1, :], prev_out), dim=2)
                # pdb.set_trace()
                out, (hh, cc) = self.ar_lstm(out, (hh, cc))
                out = self.FC(out)
                current_out = F.log_softmax(out, dim=2)
                prev_out = torch.argmax(current_out, dim=2)
                prev_out = self.class_embedding(prev_out)
                total_result[:, i:i + 1, :] = current_out
        return total_result

    def elementwise(self, fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return (h, c)



"""
     (GNN/LSTM)_(0/1/2/3)_(fc/lstm/gnn/ar4)
"""
class seq2seq(nn.Module):
    def __init__(self, encoder_type, freeze_layers, decoder):
        super(seq2seq, self).__init__()

        if encoder_type == "softLSTM":
            self.encoder = encoder_pretrained(arq="LSTM", freeze_layers=freeze_layers, soft=True)
        elif encoder_type == "LSTM":
            self.encoder = encoder_pretrained(arq="LSTM", freeze_layers=freeze_layers, soft=False)
        elif encoder_type == "softGNN":
            self.encoder = encoder_pretrained(arq="GNN", freeze_layers=freeze_layers, soft=True)
        elif encoder_type == "GNN":
            self.encoder = encoder_pretrained(arq="GNN", freeze_layers=freeze_layers, soft=False)

        if decoder == "fc":
            self.decoder = FC_pretrained(encoder_type)
        elif decoder == "lstm":
            self.decoder = lstm_decoder(32 if "GNN" in encoder_type else 64)
        elif decoder == "gnn":
            self.decoder = gnn_decoder(32 if "GNN" in encoder_type else 64)
        elif "ar" in decoder:
            teacher_forcing_ratio = float(decoder[2]) / 10
            print(f"teacher_forcing_ratio = {teacher_forcing_ratio}")
            self.decoder = AR_decoder(32 if "GNN" in encoder_type else 64, teacher_forcing_ratio)

    def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
        x = self.encoder(notes, onsets, durations, x_lengths, edge_list, fingers=None)
        x = self.decoder(x, x_lengths, edge_list, fingers)
        return x

