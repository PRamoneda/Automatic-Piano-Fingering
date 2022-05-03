import pdb

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os

GRAPH_KEYS = ['onset', 'next']
N_EDGE_TYPE = len(GRAPH_KEYS) * 2
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def edges_to_matrix(edges, num_notes, graph_keys=GRAPH_KEYS):
    if len(graph_keys) == 0:
        return None
    num_keywords = len(graph_keys)
    graph_dict = {key: i for i, key in enumerate(graph_keys)}
    if 'rest_as_forward' in graph_dict:
        graph_dict['rest_as_forward'] = graph_dict['forward']
        num_keywords -= 1
    matrix = np.zeros((num_keywords * 2, num_notes, num_notes))
    edg_indices = [(graph_dict[edg[2]], edg[0], edg[1])
                   for edg in edges
                   if edg[2] in graph_dict]
    reverse_indices = [(edg[0] + num_keywords, edg[2], edg[1]) if edg[0] != 0 else
                       (edg[0], edg[2], edg[1]) for edg in edg_indices]
    edg_indices = np.asarray(edg_indices + reverse_indices)

    matrix[edg_indices[:, 0], edg_indices[:, 1], edg_indices[:, 2]] = 1

    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = torch.Tensor(matrix)
    return matrix


class GatedGraph(nn.Module):
    def __init__(self, size, num_edge_types, secondary_size=0):
        super(GatedGraph, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if secondary_size == 0:
            secondary_size = size
        self.size = size
        self.secondary_size = secondary_size
        self.num_type = num_edge_types

        self.ba = torch.nn.Parameter(torch.Tensor(num_edge_types, size))
        self.bw = torch.nn.Parameter(torch.Tensor(secondary_size * 3))
        self.wz_wr_wh = torch.nn.Parameter(torch.Tensor(num_edge_types, size, secondary_size * 3))
        self.uz_ur = torch.nn.Parameter(torch.Tensor(size, secondary_size * 2))
        self.uh = torch.nn.Parameter(torch.Tensor(secondary_size, secondary_size))

        std_a = (2 / (secondary_size + secondary_size)) ** 0.5
        std_b = (2 / (size + secondary_size)) ** 0.5
        std_c = (2 / (size * self.num_type + secondary_size)) ** 0.5

        nn.init.normal_(self.wz_wr_wh, std=std_c)
        nn.init.normal_(self.uz_ur, std=std_b)
        nn.init.normal_(self.uh, std=std_a)
        nn.init.zeros_(self.ba)
        nn.init.zeros_(self.bw)

    def forward(self, input, edge_matrix, iteration=1):
        '''
        input (torch.Tensor): N (num_batch) x T (num_notes) x C 
        edge_matrix (torch.Tensor): N x E (edge_type) x T  x T
        
        '''
        # max_len = max([edge.shape[1] for edge in edge_matrix])
        # pdb.set_trace()
        # padded_edges = []
        # for edge in edge_matrix:
        #     padded_edge = F.pad(edge, [0, max_len - edge.shape[1], 0, max_len - edge.shape[1], 0, 0], mode='constant')
        #     print(edge.shape, padded_edge.shape, [0, 0, 0, max_len - edge.shape[1], 0, max_len - edge.shape[1]])
        #     padded_edges.append(padded_edge)
        # pdb.set_trace()
        # edge_matrix = torch.stack(padded_edges, dim=0).to(self.device)

        len_b = len(input)
        len_n = input.shape[1]
        # pdb.set_trace()
        input_broadcast = input.repeat(1, self.num_type, 1).view(-1, edge_matrix.shape[2],
                                                                 input.shape[2])  # (N x E) x T x C
        edge_batch_flatten = edge_matrix.view(len_b * self.num_type, edge_matrix.shape[-2],
                                              edge_matrix.shape[-1])  # (N x E) x T x T

        for i in range(iteration):
            activation = torch.matmul(edge_batch_flatten.transpose(1, 2), input_broadcast)  # (N x E) x T x C
            activation += self.ba.unsqueeze(1).repeat(len_b, 1, 1)
            broadcasted_wzrh = self.wz_wr_wh.unsqueeze(0).repeat(len_b, 1, 1, 1).view(activation.shape[0],
                                                                                      activation.shape[2],
                                                                                      self.wz_wr_wh.shape[-1])
            activation_wzrh = torch.bmm(activation, broadcasted_wzrh) + self.bw
            activation_wzrh = activation_wzrh.view(len_b, self.num_type, len_n, -1)

            activation_wz, activation_wr, activation_wh = torch.split(activation_wzrh, self.secondary_size, dim=-1)
            activation_wz = activation_wz.sum(1)
            activation_wr = activation_wr.sum(1)
            activation_wh = activation_wh.sum(1)

            input_uzr = torch.matmul(input, self.uz_ur)
            input_uz, input_ur = torch.split(input_uzr, self.secondary_size, dim=-1)
            temp_z = torch.sigmoid(activation_wz + input_uz)
            temp_r = torch.sigmoid(activation_wr + input_ur)

            if self.secondary_size == self.size:
                temp_hidden = torch.tanh(
                    activation_wh + torch.matmul(temp_r * input, self.uh))
                input = (1 - temp_z) * input + temp_z * temp_hidden
            else:
                temp_hidden = torch.tanh(
                    activation_wh + torch.matmul(temp_r * input[:, :, -self.secondary_size:], self.uh))
                temp_result = (1 - temp_z) * input[:, :, -self.secondary_size:] + temp_z * temp_hidden
                input = torch.cat((input[:, :, :-self.secondary_size], temp_result), 2)
        return input[:, :, -self.secondary_size:]
