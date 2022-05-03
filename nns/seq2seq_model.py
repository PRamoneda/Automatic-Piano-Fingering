import pdb
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as packer, pad_packed_sequence as padder

from nns import GGCN
from nns.GGCN import GatedGraph


class gnn_encoder(nn.Module):
    def __init__(self, input_size):
        super(gnn_encoder, self).__init__()
        self.gnn1 = GatedGraph(size=input_size, secondary_size=input_size, num_edge_types=GGCN.N_EDGE_TYPE)
        self.gnn2 = GatedGraph(size=input_size, secondary_size=input_size, num_edge_types=GGCN.N_EDGE_TYPE)
        self.gnn3 = GatedGraph(size=input_size, secondary_size=input_size, num_edge_types=GGCN.N_EDGE_TYPE)

    def forward(self, x, x_lengths, edge_list):
        x = self.gnn1(x, edge_list)
        x = self.gnn2(x, edge_list)
        x = self.gnn3(x, edge_list)
        return x


class lstm_encoder(nn.Module):
    # the encoder / feature extractor / generator? xd
    def __init__(self, input, dropout=0.0):
        super(lstm_encoder, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input, hidden_size=32, batch_first=True, num_layers=3, bidirectional=True,
                            dropout=dropout)

    def forward(self, x, x_lengths, edge_list=None):
        # pdb.set_trace()
        x_packed = packer(x.float(), x_lengths.cpu().numpy(), batch_first=True)
        # pdb.set_trace()
        output, _ = self.rnn1(x_packed.float())
        output_padded, _ = padder(output, batch_first=True)
        # only take the reverse time direction output (which have the time direction as input)
        # pdb.set_trace()
        return output_padded

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


class AR_decoder(nn.Module):
    def __init__(self, in_size):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super(AR_decoder, self).__init__()
        # parameters
        self.in_size = in_size
        self.hidden_size = 64
        self.num_layers = 1
        # nn
        self.ar_lstm = nn.LSTM(
            input_size=self.in_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.FC = nn.Linear(self.hidden_size, 5)
        self.class_embedding = nn.Embedding(5, in_size)

    def forward(self, x, x_lengths, edge_list, fingers=None, ratio_teaching_forcing=0.7):
        teaching_forcing = False
        if fingers is not None:
            teaching_forcing = random.random() < ratio_teaching_forcing

        if teaching_forcing:
            # print("teaching forcing")
            fingers = torch.unsqueeze(fingers, dim=2)
            prev_gt = torch.cat(
                (torch.zeros((fingers.shape[0], 1, fingers.shape[2]), device=self.device, dtype=torch.long),
                 fingers[:, :-1, :].type(torch.LongTensor).to(self.device)),
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

    def forward_intermittent(self, x, x_lengths, edge_list, fingers=None):
        # pdb.set_trace()
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
            if fingers is not None:
                mask = (fingers[:, i] != -1)  # mask -1 values
                prev_out[mask] = fingers[:, i].view(-1, 1)[mask]  # assign teaching forcing to not -1 values
                # pdb.set_trace()

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

    def decode_with_beam_search(self, x, x_lengths, edge_list, beam_k=20):
        '''
        
        Out
        candidates: A list of list. Each list in the candidates is a sequence of fingering in integer values
        '''
        assert x.shape[0] == 1 # Currently only works for batch size 1 

        hidden_states = self.init_hidden(1) # instead of batch, track hidden states of each beam case
        beam_embeddings = torch.zeros((1, 1, self.in_size)).to(self.device) # prev_out_embedding of each beam case
        cum_prob = torch.ones(1).to(self.device) # cumulative probability of each beam case
        candidates = [ [] ] # decoded fingering sequence of each beam case

        for i in range(x.shape[1]):
            x_repeated = x[:, i:i+1].repeat(len(beam_embeddings), 1, 1)
            cat_input = torch.cat([x_repeated, beam_embeddings], dim=2)
            out, hidden_states = self.ar_lstm(cat_input, hidden_states)
            out = self.FC(out)
            out_prob = F.softmax(out, dim=2) # use softmax to get probability

            new_cum_prob = cum_prob.unsqueeze(1) * out_prob[:,0] # calculate new cumulative probs for each of new possible cases

            prob_flatten = new_cum_prob.view(-1) 
            _, sorted_indices = prob_flatten.sort(descending=True) 

            selected_beam_ids = sorted_indices[:beam_k] # find the case id with maximum cum_prob with top_k
            cum_prob = prob_flatten[selected_beam_ids]

            prev_beam_indices = torch.div(selected_beam_ids, 5, rounding_mode='floor') # find the parent beam id for each of new selected cases
            pred_beam_fingerings = selected_beam_ids % 5 

            candidates = [candidates[prev_beam_indices[i]] + [pred_beam_fingerings[i].item()] for i in range(len(selected_beam_ids))]

            hidden_states = (hidden_states[0][:,prev_beam_indices], hidden_states[1][:,prev_beam_indices] ) # this is for LSTM hidden cell
            beam_embeddings = self.class_embedding(pred_beam_fingerings).unsqueeze(1)
            cum_prob = cum_prob / max(cum_prob)
        return candidates


class linear_decoder(nn.Module):
    def __init__(self):
        super(linear_decoder, self).__init__()
        self.FC = nn.Linear(64, 5)

    def forward(self, x, x_lengths, edge_list, fingers=None):
        return F.softmax(self.FC(x), dim=2)

    def forward_intermittent(self, x, x_lengths, edge_list, fingers=None):
        return F.softmax(self.FC(x), dim=2)


class seq2seq(nn.Module):
    def __init__(self, embedding, encoder, decoder):
        super(seq2seq, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
        # next forward parameter
        x = self.embedding(notes=notes, onsets=onsets, durations=durations, x_lengths=x_lengths)
        x = self.encoder(x=x, x_lengths=x_lengths, edge_list=edge_list)
        logits = self.decoder(x=x, x_lengths=x_lengths, edge_list=edge_list, fingers=fingers)
        return logits

    def forward_intermittent(self, notes, onsets, durations, x_lengths, edge_list, fingers=None):
        # next forward parameter
        x = self.embedding(notes=notes, onsets=onsets, durations=durations, x_lengths=x_lengths)
        x = self.encoder(x=x, x_lengths=x_lengths, edge_list=edge_list)
        logits = self.decoder.forward_intermittent(x=x, x_lengths=x_lengths, edge_list=edge_list, fingers=fingers)
        return logits

    def decode_with_beam(self, notes, onsets, durations, x_lengths, edge_list, fingers=None, beam_k=10):
        x = self.embedding(notes=notes, onsets=onsets, durations=durations, x_lengths=x_lengths)
        x = self.encoder(x=x, x_lengths=x_lengths, edge_list=edge_list)
        candidates = self.decoder.decode_with_beam_search(x=x, x_lengths=x_lengths, edge_list=edge_list, beam_k=beam_k)
        return candidates

    def freeze(self, freeze_type):
        print(f"freeze_type {freeze_type}")
        if freeze_type == '1':
            print("freeze encoder")
            for param in self.embedding.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif freeze_type == '0':
            # pass
            print("only")
