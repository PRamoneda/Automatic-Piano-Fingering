import os
import pdb
from statistics import mean

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from torch.utils.data import BatchSampler, RandomSampler

import common_classification as common
from common_classification import SoftNLLLoss, compute_training, compute_training_eval, \
    start_logging, compute_results_classification, load_model, save_model, collate_fn, get_representation
from nns.GGCN import edges_to_matrix


logging = True


def collate_fn_seq2seq(batch):
    # TODO legnth is not ok
    # collatefunction for handling with the recurrencies
    notes, onsets, durations, fingers, ids, lengths, edges = zip(*batch)

    # order by length
    notes, onsets, durations, fingers, ids, lengths, edges = \
        map(list, zip(*sorted(zip(notes, onsets, durations, fingers, ids, lengths, edges), key=lambda a: a[5], reverse=True)))
    #  pad sequences
    notes = torch.nn.utils.rnn.pad_sequence(notes, batch_first=True)
    onsets = torch.nn.utils.rnn.pad_sequence(onsets, batch_first=True)
    durations = torch.nn.utils.rnn.pad_sequence(durations, batch_first=True)
    fingers_padded = torch.nn.utils.rnn.pad_sequence(fingers, batch_first=True, padding_value=-1)
    edge_list = []
    # print("notes", len(notes), len(notes[0]), lengths)
    for e, le in zip(edges, lengths):
        edge_list.append(edges_to_matrix(e, le))
    max_len = max([edge.shape[1] for edge in edge_list])
    new_edges = torch.stack(
        [
            F.pad(edge, (0, max_len - edge.shape[1], 0, max_len - edge.shape[1], 0, 0), mode='constant')
            for edge in edge_list
        ]
    , dim=0)
    # If a vector input was given for the sequences, expand (B x T_max) to (B x T_max x 1)
    if notes.ndim == 2:
        notes.unsqueeze_(2)
        onsets.unsqueeze_(2)
        durations.unsqueeze_(2)
    return notes, onsets, durations, fingers_padded, ids, torch.IntTensor(lengths), new_edges


def create_dataset(representation, batch_size_training=64):
    test_rh, test_lh, noisy_validation_rh, noisy_validation_lh, noisy_windowed = get_representation(representation)
    test_rh_loader = common.create_loader(test_rh, 5, num_workers=1, batch_size=1, collate_fn=collate_fn_seq2seq)
    test_lh_loader = common.create_loader(test_lh, 6, num_workers=1, batch_size=1, collate_fn=collate_fn_seq2seq)
    noisy_validation_rh_loader = common.create_loader(noisy_validation_rh, 0, num_workers=1, batch_size=1, collate_fn=collate_fn_seq2seq)
    noisy_validation_lh_loader = common.create_loader(noisy_validation_lh, 0, num_workers=1, batch_size=1, collate_fn=collate_fn_seq2seq)
    noisy_windowed_loader = common.create_loader_augmented(noisy_windowed, 0, num_workers=4, batch_size=batch_size_training, collate_fn=collate_fn_seq2seq)
    print(len(noisy_windowed_loader))
    return test_rh_loader, test_lh_loader, noisy_validation_rh_loader, noisy_validation_lh_loader, noisy_windowed_loader


def training_loop(data, device, model, args):
    test_rh, test_lh, noisy_validation_rh, noisy_validation_lh, noisy_windowed = data

    n_epochs = 2000
    best_acc = 0
    smoothing_rate = 0.3
    patience, trials = 100, 0
    lr = .00005

    if logging:
        writer = start_logging(args, lr, n_epochs, patience)
    else:
        writer = None

    print(f'learning rate = {lr}')

    model = model.to(device)

    if 'soft' in args['architecture']:
        criterion = SoftNLLLoss(device, smoothing_rate)
        print("softening labels!")
    else:
        criterion = nn.NLLLoss(ignore_index=-1)
        print("usual loss!")

    if 'weight_decay' in args['architecture']:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        print("with weight decay!")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    print('Start model training (noisy seq2seq version)')
    for epoch in range(1, n_epochs + 1):
        running_loss = []

        # print(len(noisy_windowed))
        for i, (notes, onsets, durations, fingers, ids, lengths, edge_list) in enumerate(noisy_windowed):
            # print(i)
            notes = notes.to(device)
            onsets = onsets.to(device)
            durations = durations.to(device)
            fingers = fingers.to(device)
            lengths = lengths.to(device)
            edge_list = edge_list.to(device)
            model.train()
            opt.zero_grad()
            # print(f"{i} kk1")
            # pdb.set_trace()

            out = model.forward_intermittent(notes, onsets, durations, lengths, edge_list, fingers)
            loss = criterion(out.transpose(1, 2), fingers)
            loss.backward()
            opt.step()
            running_loss.append(loss.item())
        print("validation")

        acc_rh = common.compute_results_seq2seq(args, noisy_validation_rh, model, device, writer, logging, False)
        acc_lh = common.compute_results_seq2seq(args, noisy_validation_lh, model, device, writer, logging, False)
        acc = (acc_rh + acc_lh) / 2
        print(f"Validation (General match rate): rh:{acc_rh:2.2%} lh:{acc_lh:2.2%}")

        cheating_acc_rh = common.compute_results_seq2seq(args, test_rh, model, device, writer, logging, False)
        cheating_acc_lh = common.compute_results_seq2seq(args, test_lh, model, device, writer, logging, False)
        print(f"Test (General match rate): rh:{cheating_acc_rh:2.2%} lh:{cheating_acc_lh:2.2%}")

        if logging:
            writer.add_scalar("train/cheating_lh", cheating_acc_rh, epoch)
            writer.add_scalar("train/cheating_rh", cheating_acc_lh, epoch)

        if logging:
            # log loss
            writer.add_scalar("train_noisy/loss", mean(running_loss), epoch)
            # log evaluation in noisy dataset
            writer.add_scalar("eval_noisy/acc", acc, epoch)
            # log evaluation in nakamura dataset

        # Early stopping
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {mean(running_loss):.4f}. Acc.: {acc:2.2%}')
        if acc > best_acc:
            trials = 0
            best_acc = acc
            save_model(f'models/best_{os.path.basename("#".join([x for x in args.values()]))}.pth',
                       epoch, model, opt, criterion)
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience and epoch >= 70:
                print(f'Early stopping on epoch {epoch}')
                break
    print(f"Validation (General match rate):: {best_acc:2.2%}")
    return writer, logging


def run_test(data, device, model, args, writer):
    test_rh, test_lh, noisy_validation_rh, noisy_validation_lh, noisy_windowed = data
    model, _, _, _ = load_model(f'models/best_{os.path.basename("#".join([x for x in args.values()]))}.pth', model)
    model.to(device)
    acc_rh = common.compute_results_seq2seq(args, test_rh, model, device, writer, logging)
    acc_lh = common.compute_results_seq2seq(args, test_lh, model, device, writer, logging)
    print(f"Test (General match rate): rh:{acc_rh:2.2%} lh:{acc_lh:2.2%}")


if __name__ == '__main__':
    pass
