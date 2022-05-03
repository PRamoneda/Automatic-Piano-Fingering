import os
import pdb
from statistics import mean

import torch
from torch import nn
from sklearn.metrics import accuracy_score

import common_classification as common
from common_classification import SoftNLLLoss, compute_training, \
    start_logging, compute_results_classification, load_model, save_model, get_representation
import torch.nn.functional as F

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


def create_dataset(representation):
    train, test_rh, test_lh = get_representation(representation)

    train_loader = common.create_loader_augmented(train, 0, num_workers=2, batch_size=3, collate_fn=collate_fn_seq2seq)
    test_rh_loader = common.create_loader(test_rh, 5, num_workers=1, batch_size=1, collate_fn=collate_fn_seq2seq)
    test_lh_loader = common.create_loader(test_lh, 6, num_workers=1, batch_size=1, collate_fn=collate_fn_seq2seq)

    return test_rh_loader, test_lh_loader, train_loader


def training_loop(data, device, model, args):
    test_rh, test_lh, train = data

    n_epochs = 1000
    best_acc_rh = 0
    best_acc_lh = 0

    patience, trials = 100, 0
    lr = .0005
    if logging:
        writer = start_logging(args, lr, n_epochs, patience)
    else:
        writer = None

    print(f'learning rate = {lr}')

    model = model.to(device)


    criterion = nn.NLLLoss(ignore_index=-1)
    print("usual loss!")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print('Start generalization experiment!!')
    for epoch in range(1, n_epochs + 1):
        running_loss = []
        for i, (notes, onsets, durations, fingers, ids, lengths, edge_list) in enumerate(train):
            notes = notes.to(device)
            onsets = onsets.to(device)
            durations = durations.to(device)
            fingers = fingers.to(device)
            lengths = lengths.to(device)
            edge_list = edge_list.to(device)
            model.train()
            opt.zero_grad()
            out = model(notes, onsets, durations, lengths, edge_list, fingers)

            loss = criterion(out.transpose(1, 2), fingers)
            loss.backward()
            opt.step()
            running_loss.append(loss.item())

        acc_rh = common.compute_results_seq2seq(args, test_rh, model, device, writer, logging, save=False)
        acc_lh = common.compute_results_seq2seq(args, test_lh, model, device, writer, logging, save=False)
        print(f"Test (General match rate): rh:{acc_rh:2.2%} lh:{acc_lh:2.2%}")

        if logging:

            writer.add_scalar("test/lh", acc_lh, epoch)
            writer.add_scalar("test/rh", acc_rh, epoch)

        print(f"Loss = {mean(running_loss)}")
        if logging:
            # log loss
            writer.add_scalar("train/loss", mean(running_loss), epoch)

        # Early stopping
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {mean(running_loss):.4f}. Acc: (rh:{acc_rh:2.2%} lh:{acc_lh:2.2%})')
        if acc_rh > best_acc_rh or acc_lh > best_acc_lh :
            if acc_rh > best_acc_rh:
                best_acc_rh = acc_lh
                save_model(f'models/best_rh_{os.path.basename("#".join([x for x in args.values()]))}.pth',
                           epoch, model, opt, criterion)
            if acc_lh > best_acc_lh:
                best_acc_lh = acc_lh
                save_model(f'models/best_lh_{os.path.basename("#".join([x for x in args.values()]))}.pth',
                           epoch, model, opt, criterion)
            trials = 0

            print(f'Epoch {epoch} best model saved with accuracy: (rh:{acc_rh:2.2%} lh:{acc_lh:2.2%})')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
    print(f"Validation (General match rate): (rh:{best_acc_rh:2.2%} lh:{best_acc_lh:2.2%}) ")
    return writer, logging


def run_test(data, device, model, args, writer):
    test_rh, test_lh, _ = data

    model, _, _, _ = load_model(f'models/best_rh_{os.path.basename("#".join([x for x in args.values()]))}.pth', model)
    model.to(device)
    test_acc_rh = common.compute_results_seq2seq(args, test_rh, model, device, writer, logging)

    model, _, _, _ = load_model(f'models/best_lh_{os.path.basename("#".join([x for x in args.values()]))}.pth', model)
    model.to(device)
    test_acc_lh = common.compute_results_seq2seq(args, test_lh, model, device, writer, logging)

    print(f"Test (General match rate): rh:{test_acc_rh:2.2%} lh:{test_acc_lh:2.2%}")


if __name__ == '__main__':
    pass
