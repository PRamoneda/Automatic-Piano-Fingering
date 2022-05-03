import os
import pdb
import random
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm

import metrics
from loader import load_pfd_right_w11, load_pfd_right_noisy_w11, salami_tensor, filter_edges, compute_edge_list_window, \
    load_pfd_full_augmented_w11, load_noisy_full_augmented_w11, load_pfd_seq2seq_w11, load_noisy_random_seq2seq, \
    load_nakamura_augmented_seq2seq_merged, load_validation_experiment, load_generalization, \
    load_no_augmented_noisy_random_seq2seq, load_nakamura_augmented_seq2seq_no_merged, \
    load_nakamura_no_augmented_seq2seq
from nns.GGCN import edges_to_matrix
from utils import save_json, load_json


class SoftNLLLoss(nn.Module):
    def __init__(self, device, smoothing=0.3, ignore_index=-1):
        super(SoftNLLLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.device = device
        self.ignore_index = ignore_index

    def forward(self, logprobs, y):
        # previously computed ... p = F.log_softmax(y_hat, 1)
        mask = (y.unsqueeze(1) != self.ignore_index)
        logprobs = logprobs * mask
        y_masked = y * mask.squeeze(1)
        nll_loss = -logprobs.gather(dim=1, index=y_masked.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def collate_fn(batch):
    notes, onsets, durations, fingers, ids, lengths, edges = zip(*batch)

    notes = torch.unsqueeze(torch.stack(notes), dim=2)
    onsets = torch.unsqueeze(torch.stack(onsets), dim=2)
    durations = torch.unsqueeze(torch.stack(durations), dim=2)
    fingers = torch.stack(fingers)
    return notes.float(), onsets, durations, fingers, ids, torch.IntTensor(lengths), edges


def get_representation(representation):
    if representation == 'right_full_w11':
        data = load_pfd_right_w11()
    if representation == 'right_noisy_w11':
        data = load_pfd_right_noisy_w11()
    if representation == 'full_augmented_w11':
        data = load_pfd_full_augmented_w11()
    if representation == 'augmented_noisy_w11':
        data = load_noisy_full_augmented_w11()
    if representation == 'augmented_seq2seq':
        data = load_pfd_seq2seq_w11()
    if representation == 'augmented_noisy_random_seq2seq':
        data = load_noisy_random_seq2seq()
    if representation == 'no_augmented_noisy_random_seq2seq':
        data = load_no_augmented_noisy_random_seq2seq()
    if representation == 'generalization':
        data = load_generalization()
    if representation == 'nakamura_augmented_seq2seq_merged':
        data = load_nakamura_augmented_seq2seq_merged()
    if representation == 'nakamura_no_augmented_seq2seq_separated':
        data = load_nakamura_no_augmented_seq2seq()
    if representation == 'nakamura_augmented_seq2seq_separated':
        data = load_nakamura_augmented_seq2seq_no_merged()
    if 'validation_experiment' in representation:
        data = load_validation_experiment(representation)
    return data


def create_loader(data, subset, num_workers=1, batch_size=1, collate_fn=collate_fn):
    dataset = fingering_subset(data)
    if batch_size is None:
        batch_size = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size,
                                                  num_workers=num_workers, pin_memory=True)
    loader.subset = subset

    return loader


def create_loader_augmented(data, subset, num_workers=1, batch_size=3, collate_fn=collate_fn):
    dataset = fingering_subset_augmented(data)
    print("batchsize", batch_size)
    if batch_size is None:
        batch_size = len(dataset)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size,
                                         num_workers=num_workers, pin_memory=True, shuffle=True)

    loader.subset = subset

    return loader


class fingering_subset(torch.utils.data.Dataset):
    def __init__(
            self,
            data,
    ):

        self.list_notes, \
        self.list_onsets, \
        self.list_durations, \
        self.list_fingers, \
        self.list_ids, \
        self.list_lengths, \
        self.list_edges = data

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        notes = torch.from_numpy(self.list_notes[index].astype(float))
        onsets = torch.from_numpy(self.list_onsets[index].astype(np.float32))
        durations = torch.from_numpy(self.list_durations[index].astype(np.float32))
        fingers = torch.from_numpy(self.list_fingers[index].astype(np.compat.long))
        ids = self.list_ids[index]
        length = self.list_lengths[index]
        edges = self.list_edges[index]
        return notes, onsets, durations, fingers, ids, length, edges

    def __len__(self):
        return len(self.list_notes)


class fingering_subset_augmented(torch.utils.data.Dataset):
    subset = 0

    def __init__(
            self,
            data,
    ):
        self.windowed_data = data

    def __getitem__(self, index):
        choice_index = random.choice(range(len(self.windowed_data[index])))
        # if index == 0:
        #     print("index 0", choice_index)
        note_windowed, \
        onset_windowed, \
        duration_windowed, \
        finger_windowed, \
        ids_windowed, \
        lengths_windowed, \
        edge_windowed = self.windowed_data[index][choice_index]

        notes = torch.from_numpy(note_windowed.astype(float))
        onsets = torch.from_numpy(onset_windowed.astype(np.float32))
        durations = torch.from_numpy(duration_windowed.astype(np.float32))
        fingers = torch.from_numpy(finger_windowed.astype(np.compat.long))
        ids = ids_windowed
        length = lengths_windowed
        edges = edge_windowed
        return notes, onsets, durations, fingers, ids, length, edges

    def __len__(self):
        return len(self.windowed_data)


# class pfd_dataset(torch.utils.data.Dataset):
#     def __init__(
#             self,
#             data,
#             subset=0,
#             augmented=False
#     ):
#         """
#         """
#         self.subset = subset
#         self.augmented = augmented
#
#         train_data, val_data, test_data, windowed_data, noisy_train_data, noisy_validation_data = data
#
#         self.note_train, \
#         self.onset_train, \
#         self.duration_train, \
#         self.finger_train, \
#         self.ids_train, \
#         self.lengths_train, \
#         self.edge_train = train_data
#
#         self.note_val, \
#         self.onset_val, \
#         self.duration_val, \
#         self.finger_val, \
#         self.ids_val, \
#         self.lengths_val, \
#         self.edge_val = val_data
#
#         self.note_test, \
#         self.onset_test, \
#         self.duration_test, \
#         self.finger_test, \
#         self.ids_test, \
#         self.lengths_test, \
#         self.edge_test = test_data
#
#         if not augmented:
#             self.note_windowed, \
#             self.onset_windowed, \
#             self.duration_windowed, \
#             self.finger_windowed, \
#             self.ids_windowed, \
#             self.lengths_windowed, \
#             self.edge_windowed = windowed_data
#         else:
#             self.windowed_data = windowed_data
#
#         self.note_noisy_train, \
#         self.onset_noisy_train, \
#         self.duration_noisy_train, \
#         self.finger_noisy_train, \
#         self.ids_noisy_train, \
#         self.lengths_noisy_train, \
#         self.edge_noisy_train = noisy_train_data
#
#         self.note_noisy_validation, \
#         self.onset_noisy_validation, \
#         self.duration_noisy_validation, \
#         self.finger_noisy_validation, \
#         self.ids_noisy_validation, \
#         self.lengths_noisy_validation, \
#         self.edge_noisy_validation = noisy_validation_data
#
#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#         if self.subset == 0:
#             notes = torch.from_numpy(self.note_train[index].astype(float))
#             onsets = torch.from_numpy(self.onset_train[index].astype(np.float32))
#             durations = torch.from_numpy(self.duration_train[index].astype(np.float32))
#             fingers = torch.from_numpy(self.finger_train[index].astype(np.compat.long))
#             ids = self.ids_train[index]
#             length = self.lengths_train[index]
#             edges = self.edge_train[index]
#         elif self.subset == 1:
#             notes = torch.from_numpy(self.note_val[index].astype(float))
#             onsets = torch.from_numpy(self.onset_val[index].astype(np.float32))
#             durations = torch.from_numpy(self.duration_val[index].astype(np.float32))
#             fingers = torch.from_numpy(self.finger_val[index].astype(np.compat.long))
#             ids = self.ids_val[index]
#             length = self.lengths_val[index]
#             edges = self.edge_val[index]
#         elif self.subset == 2:  # self.set == 2:
#             notes = torch.from_numpy(self.note_test[index].astype(float))
#             onsets = torch.from_numpy(self.onset_test[index].astype(np.float32))
#             durations = torch.from_numpy(self.duration_test[index].astype(np.float32))
#             fingers = torch.from_numpy(self.finger_test[index].astype(np.compat.long))
#             ids = self.ids_test[index]
#             length = self.lengths_test[index]
#             edges = self.edge_test[index]
#
#         elif self.subset == 4:
#             if self.augmented:
#                 choice_index = random.choice(range(len(self.windowed_data[index])))
#                 # if index == 0:
#                 #     print("index 0", choice_index)
#                 note_windowed, \
#                 onset_windowed, \
#                 duration_windowed, \
#                 finger_windowed, \
#                 ids_windowed, \
#                 lengths_windowed, \
#                 edge_windowed = self.windowed_data[index][choice_index]
#
#                 notes = torch.from_numpy(note_windowed.astype(float))
#                 onsets = torch.from_numpy(onset_windowed.astype(np.float32))
#                 durations = torch.from_numpy(duration_windowed.astype(np.float32))
#                 fingers = torch.from_numpy(finger_windowed.astype(np.compat.long))
#                 ids = ids_windowed
#                 length = lengths_windowed
#                 edges = edge_windowed
#             else:
#                 notes = torch.from_numpy(self.note_windowed[index].astype(float))
#                 onsets = torch.from_numpy(self.onset_windowed[index].astype(np.float32))
#                 durations = torch.from_numpy(self.duration_windowed[index].astype(np.float32))
#                 fingers = torch.from_numpy(self.finger_windowed[index].astype(np.compat.long))
#                 ids = self.ids_windowed[index]
#                 length = self.lengths_windowed[index]
#                 edges = self.edge_windowed[index]
#
#         elif self.subset == 5:
#             notes = torch.from_numpy(self.note_noisy_train[index].astype(float))
#             onsets = torch.from_numpy(self.onset_noisy_train[index].astype(np.float32))
#             durations = torch.from_numpy(self.duration_noisy_train[index].astype(np.float32))
#             fingers = torch.from_numpy(self.finger_noisy_train[index].astype(np.compat.long))
#             ids = self.ids_noisy_train[index]
#             length = self.lengths_noisy_train[index]
#             edges = self.edge_noisy_train[index]
#
#         elif self.subset == 6:
#             notes = torch.from_numpy(self.note_noisy_validation[index].astype(float))
#             onsets = torch.from_numpy(self.onset_noisy_validation[index].astype(np.float32))
#             durations = torch.from_numpy(self.duration_noisy_validation[index].astype(np.float32))
#             fingers = torch.from_numpy(self.finger_noisy_validation[index].astype(np.compat.long))
#             ids = self.ids_noisy_validation[index]
#             length = self.lengths_noisy_validation[index]
#             edges = self.edge_noisy_validation[index]
#         return notes, onsets, durations, fingers, ids, length, edges
#
#     def __len__(self):
#         if self.subset == 0:
#             return len(self.note_train)
#         elif self.subset == 1:
#             return len(self.note_val)
#         elif self.subset == 4:
#             if not self.augmented:
#                 return len(self.note_windowed)
#             else:
#                 return len(self.windowed_data)
#         elif self.subset == 5:
#             return len(self.note_noisy_train)
#         elif self.subset == 6:
#             return len(self.note_noisy_validation)
#         elif self.subset == 2:
#             return len(self.note_test)



def compute_edges_batch(edge_list, lengths):
    edges = []
    for e, le in zip(edge_list, lengths):
        edges.append(edges_to_matrix(e, le))
    new_edges = torch.stack(edges, dim=0)
    return new_edges


def compute_training(loader, device, model, opt, criterion, data_quality, running_loss):
    window_trues, window_preds = [], []
    for i, (notes, onsets, durations, fingers, ids, lengths, edge_list) in enumerate(loader):
        # input data to device
        edge_list = compute_edges_batch(edge_list, lengths)
        notes = notes.to(device)
        onsets = onsets.to(device)
        durations = durations.to(device)
        fingers = fingers.to(device)
        lengths = lengths.to(device)
        edge_list = edge_list.to(device)
        # print(notes.cpu())
        # training
        model.train()
        out = model(notes, onsets, durations, lengths, edge_list)

        window_trues.extend(fingers[:, 5].cpu().tolist())
        window_preds.extend(out.argmax(dim=1).cpu().tolist())
        # pdb.set_trace()
        # print("train", notes.cpu().tolist(), out.cpu().tolist(), out.argmax(dim=1).cpu().tolist()[0])
        opt.zero_grad()
        # print(fingers[:, 5])
        loss = criterion(out, fingers[:, 5])
        loss.backward()
        opt.step()
        running_loss.append(loss.item())
    return model, opt, criterion, running_loss, window_trues, window_preds


def compute_training_eval(loader, device, model):
    model.eval()
    window_trues, window_preds = [], []
    for i, (notes, onsets, durations, fingers, ids, lengths, edge_list) in enumerate(loader):
        edge_list = compute_edges_batch(edge_list, lengths)
        notes = notes.to(device)
        onsets = onsets.to(device)
        durations = durations.to(device)
        fingers = fingers.to(device)
        lengths = lengths.to(device)
        edge_list = edge_list.to(device)

        out = model(notes, onsets, durations, lengths, edge_list)
        window_trues.extend(fingers[:, 5].cpu().tolist())
        window_preds.extend(out.argmax(dim=1).cpu().tolist())
    return window_trues, window_preds


# def compute_acc_seq2seq(loader, device, model):
#     model.eval()
#     window_trues, window_preds = [], []
#     for i, (notes, onsets, durations, fingers, ids, lengths, edge_list) in enumerate(tqdm(loader)):
#         # print(i)
#         # edge_list = compute_edges_batch(edge_list, lengths)
#         notes = notes.to(device)
#         onsets = onsets.to(device)
#         durations = durations.to(device)
#         fingers = fingers.to(device)
#         lengths = lengths.to(device)
#         # edge_list = edge_list.to(device)
#         out = model(notes, onsets, durations, lengths, edge_list, fingers=None)
#         fingers_gt = fingers.cpu().tolist()
#         fingers_pred = out.argmax(dim=2).cpu().tolist()
#         # pdb.set_trace()
#         for ii, batch in enumerate(fingers_gt):
#             for jj, f in enumerate(batch):
#                 if f != -1:
#                     window_trues.append(fingers_gt[ii][jj])
#                     window_preds.append(fingers_pred[ii][jj])
#     acc = accuracy_score(y_true=window_trues, y_pred=window_preds)
#     return acc


def start_logging(args, lr, n_epochs, patience):

    if len(sys.argv) == 3 and sys.argv[2] == "cluster":
        runs_path = "/homedtic/pramoneda/gnn_fingering/runs"
    else:
        runs_path = 'runs'

    if not os.path.exists(f"{runs_path}/{args['alias']}"):
        os.mkdir(f"{runs_path}/{args['alias']}/")
    if not os.path.exists(f"{runs_path}/{args['alias']}/{args['architecture']}"):
        os.mkdir(f"{runs_path}/{args['alias']}/{args['architecture']}")

    writer = SummaryWriter(f"{runs_path}/{args['alias']}/{args['architecture']}", comment=args['architecture'])
    params = {"learning_rate": lr, "max_epochs": n_epochs, "patience": patience}
    for k, v in params.items():
        print(k)
        writer.add_scalar("parameters/" + k, v)
    return writer


def save_compute_results(model, loader_set, args, gmr, hmr, smr, logging, writer):
    if loader_set.subset == 1:
        subset = 'train_rh'
    elif loader_set.subset == 2:
        subset = 'train_lh'
    elif loader_set.subset == 3:
        subset = 'val_rh'
    elif loader_set.subset == 4:
        subset = 'val_lh'
    elif loader_set.subset == 5:
        subset = 'test_rh'
    elif loader_set.subset == 6:
        subset = 'test_lh'
    elif loader_set.subset == 7:
        subset = 'test-fair_rh'
    elif loader_set.subset == 8:
        subset = 'test-fair_lh'

    print(f"subset = {subset}")
    if logging:
        writer.add_scalar(f"results_nakamura/{subset}/gmr", gmr)
        writer.add_scalar(f"results_nakamura/{subset}/hmr", hmr)
        writer.add_scalar(f"results_nakamura/{subset}/smr", smr)

    if len(sys.argv) == 3 and sys.argv[2] == "cluster":
        results_path = "/homedtic/pramoneda/gnn_fingering/results"
    else:
        results_path = 'results'

    json_path = f"{results_path}/{args['alias']}#{args['rep']}#{args['architecture_type']}.json"
    if not os.path.exists(json_path):
        save_json({}, json_path)
    new_json = load_json(json_path)
    if args['architecture'] not in new_json:
        print(metrics.write_number_parameters(model))
        print(model)
        new_json[args['architecture']] = {
            'params': metrics.write_number_parameters(model),

            'train_rh': {},
            'val_rh': {},
            'test_rh': {},
            'test-fair_rh': {},
            'train_lh': {},
            'val_lh': {},
            'test_lh': {},
            'test-fair_lh': {}
        }

    if 'gmr' not in new_json[args['architecture']][subset] or new_json[args['architecture']][subset]['gmr'] < gmr:
        new_json[args['architecture']][subset] = {
            'gmr': gmr,
            'hmr': hmr,
            'smr': smr
        }
    save_json(new_json, json_path)


def compute_results_seq2seq(args, loader_set, model, device, writer, logging, save=True):
    model.eval()
    preds = []
    trues = []
    total_lengths = []
    total_ids = []
    for notes, onsets, durations, fingers, ids, lengths, edge_list in loader_set:
        # pdb.set_trace()
        notes = notes.to(device)
        onsets = onsets.to(device)
        durations = durations.to(device)
        lengths = lengths.to(device)
        edge_list = edge_list.to(device)
        out = model(notes, onsets, durations, lengths, edge_list, fingers=None)
        preds.extend(out.argmax(dim=2).cpu().tolist())
        trues.extend(fingers.tolist())
        total_lengths.extend(lengths.cpu().tolist())
        total_ids.extend(ids)

    gmr = metrics.avg_general_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
    hmr = metrics.avg_highest_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
    # hand = 'left' if loader_set % 2 == 0 else 'right'
    # smr = metrics.avg_soft_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths, hand=hand)
    smr = 0
    if save:
        if args is not None:
            save_compute_results(model, loader_set, args, gmr, hmr, smr, logging, writer)
    return gmr


def compute_acc_seq2seq(args, loader_set, model, device, writer, logging, save=True):
    model.eval()
    preds = []
    trues = []
    total_lengths = []
    total_ids = []
    for notes, onsets, durations, fingers, ids, lengths, edge_list in loader_set:
        # pdb.set_trace()
        notes = notes.to(device)
        onsets = onsets.to(device)
        durations = durations.to(device)
        lengths = lengths.to(device)
        edge_list = edge_list.to(device)
        out = model(notes, onsets, durations, lengths, edge_list, fingers=None)
        preds.extend(out.argmax(dim=2).cpu().tolist())
        trues.extend(fingers.tolist())
        total_lengths.extend(lengths.cpu().tolist())
        total_ids.extend(ids)

    final_preds, final_trues = [], []
    for pred, true, ll in zip(preds, trues, total_lengths):
        final_preds.extend(pred[:ll])
        final_trues.extend(true[:ll])

    acc = accuracy_score(final_preds, final_trues)
    return acc


def compute_results_seq2seq_with_beam(args, loader_set, model, device, writer, logging, save=True, beam_k=10):
    model.eval()
    preds = []
    trues = []
    total_lengths = []
    total_ids = []
    for notes, onsets, durations, fingers, ids, lengths, edge_list in loader_set:
        # pdb.set_trace()
        notes = notes.to(device)
        onsets = onsets.to(device)
        durations = durations.to(device)
        lengths = lengths.to(device)
        edge_list = edge_list.to(device)
        # pdb.set_trace()
        candidates = model.decode_with_beam(notes, onsets, durations, lengths, edge_list, fingers=None, beam_k=beam_k)
        preds.append(candidates[0])
        trues.extend(fingers.tolist())
        total_lengths.extend(lengths.cpu().tolist())
        total_ids.extend(ids)

    gmr = metrics.avg_general_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
    hmr = metrics.avg_highest_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
    # hand = 'left' if loader_set % 2 == 0 else 'right'
    # smr = metrics.avg_soft_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths, hand=hand)
    smr = 0
    if save:
        if args is not None:
            save_compute_results(model, loader_set, args, gmr, hmr, smr, logging, writer)
    return gmr

def compute_results_classification(args, loader_set, model, device, writer, logging):
    model.eval()
    preds = []
    trues = []
    total_lengths = []
    total_ids = []
    for notes, onsets, durations, fingers, ids, lengths, _ in loader_set:
        fingers_piece = []
        for nn, o, d in zip(salami_tensor(notes, window_size=11, hop_size=1, normalize=False),
                            salami_tensor(onsets, window_size=11, hop_size=1, normalize=True),
                            salami_tensor(durations, window_size=11, hop_size=1, normalize=True)):
            e = compute_edge_list_window({'onsets': o[0, :, 0].tolist(), 'pitchs': nn[0, :, 0].tolist()})
            e = compute_edges_batch([e], [11])
            nn = nn.to(device)
            o = o.to(device)
            d = d.to(device)
            e = e.to(device)

            out = model(nn, o, d, torch.Tensor([11]).to(device), e)
            fingers_piece.append(out.argmax(dim=1).cpu().tolist()[0])

        preds.append(fingers_piece)
        trues.append(fingers.tolist()[0])
        # print("notes", len((notes[0, :, 0]*127).cpu().tolist()), (notes[0, :, 0]*127).cpu().tolist())
        # print("preds", len(fingers_piece), fingers_piece)
        # print("trues", len(fingers.tolist()[0]), fingers.tolist()[0])
        total_lengths.append(lengths.cpu().tolist()[0])
        total_ids.append(ids[0])
    # pdb.set_trace()
    gmr = metrics.avg_general_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
    hmr = metrics.avg_highest_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
    # hand = 'left' if loader_set.subset % 2 == 0 else 'right'
    # smr = metrics.avg_soft_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths, hand=hand)
    smr = 0
    if args is not None:
        save_compute_results(model, loader_set, args, gmr, hmr, smr, logging, writer)

    return gmr


def load_model(path, model, optimizer=None, device=None):
    if device is None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], map_location=device)
    epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']
    return model, optimizer, epoch, criterion


def save_model(path, epoch, model, optimizer, criterion):

    if len(sys.argv) == 3 and sys.argv[2] == "cluster":
        path = f"/homedtic/pramoneda/gnn_fingering/{path}"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion': criterion
    }, path)


if __name__ == '__main__':
    logits = F.softmax(torch.rand((64, 5, 63)), dim=1) - 2
    y = torch.randint(-1, 4, (64, 63))
    criterion = SoftNLLLoss('gpu:0')
    criterion(logits, y)




