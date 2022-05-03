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
    # train, train_rh, train_lh, val_rh, val_lh, test_rh, test_lh = get_representation('nakamura_augmented_seq2seq_separated')
    #
    # train_rh_augmented_loader = common.create_loader_augmented(train, 0, num_workers=4, batch_size=1, collate_fn=collate_fn_seq2seq)
    # train_lh_augmented_loader = common.create_loader_augmented(train, 0, num_workers=4, batch_size=1, collate_fn=collate_fn_seq2seq)
    train, train_rh, train_lh, val_rh, val_lh, test_rh, test_lh = get_representation('nakamura_no_augmented_seq2seq_separated')

    train_rh_augmented_loader = common.create_loader(train, 0, num_workers=4, batch_size=64,
                                                     collate_fn=collate_fn_seq2seq)
    train_lh_augmented_loader = common.create_loader(train, 0, num_workers=4, batch_size=64,
                                                     collate_fn=collate_fn_seq2seq)

    train_rh_loader = common.create_loader(train_rh, 1, num_workers=1, batch_size=None, collate_fn=collate_fn_seq2seq)
    train_lh_loader = common.create_loader(train_lh, 2, num_workers=1, batch_size=None, collate_fn=collate_fn_seq2seq)
    val_rh_loader = common.create_loader(val_rh, 3, num_workers=1, batch_size=None, collate_fn=collate_fn_seq2seq)
    val_lh_loader = common.create_loader(val_lh, 4, num_workers=1, batch_size=None, collate_fn=collate_fn_seq2seq)
    test_rh_loader = common.create_loader(test_rh, 5, num_workers=1, batch_size=None, collate_fn=collate_fn_seq2seq)
    test_lh_loader = common.create_loader(test_lh, 6, num_workers=1, batch_size=None, collate_fn=collate_fn_seq2seq)

    return train_rh_loader, train_lh_loader, val_rh_loader, val_lh_loader, test_rh_loader, test_lh_loader, train_rh_augmented_loader, train_lh_augmented_loader


def training_loop_each_hand(hand, train_augmented, train, validation, test, device, model, args, writer):
    n_epochs = 1000
    best_acc = 0

    patience, trials = 50, 0
    lr = .00005
    if logging and writer is None:
        writer = start_logging(args, lr, n_epochs, patience)

    print(f'learning rate = {lr}')
    model = model.to(device)

    criterion = nn.NLLLoss(ignore_index=-1)
    print("usual loss!")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    print(f'Start model finetuning seq2seq {hand}')
    for epoch in range(1, n_epochs + 1):
        running_loss = []
        for i, (notes, onsets, durations, fingers, ids, lengths, edge_list) in enumerate(train_augmented):
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


        # print(window_preds)
        train_acc = common.compute_acc_seq2seq(args, train, model, device, writer, logging, save=False)
        print(f"Train: {hand}: {train_acc:2.2%}")

        val_acc = common.compute_acc_seq2seq(args, validation, model, device, writer, logging, save=False)
        print(f"Validation: {hand}: {val_acc:2.2%}")

        test_acc = common.compute_acc_seq2seq(args, test, model, device, writer, logging, save=False)
        print(f"Test: {hand}: {test_acc:2.2%}")

        print(f"Loss = {mean(running_loss)}")

        if logging:
            writer.add_scalar(f"{hand}/train", train_acc, epoch)
            writer.add_scalar(f"{hand}/val", val_acc, epoch)
            writer.add_scalar(f"{hand}/test", test_acc, epoch)
            writer.add_scalar(f"{hand}/loss", mean(running_loss), epoch)

        # Early stopping
        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {mean(running_loss):.4f}. val Acc: (hand:{val_acc:2.2%} test ACC: (lh:{test_acc:2.2%})')

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(f'models/best_{hand}_{os.path.basename("#".join([x for x in args.values()]))}.pth',
                       epoch, model, opt, criterion)

            trials = 0

            print(f'Epoch {epoch} best model saved with accuracy {hand}: {best_acc:2.2%} ')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
    print(f"Validation: {hand}:{best_acc:2.2%}) ")
    return writer, logging


def training_loop(data, device, model, args):
    train_rh, train_lh, val_rh, val_lh, test_rh, test_lh, train_rh_augmented, train_lh_augmented = data

    writer, _ = training_loop_each_hand('rh', train_rh_augmented, train_rh, val_rh, test_rh, device, model, args, writer=None)
    writer, _ = training_loop_each_hand('lh', train_lh_augmented, train_lh, val_lh, test_lh, device, model, args, writer=writer)

    return writer, logging



def run_test(data, device, model, args, writer):
    train_rh, train_lh, val_rh, val_lh, test_rh, test_lh, _, _ = data

    model, _, _, _ = load_model(f'models/best_rh_{os.path.basename("#".join([x for x in args.values()]))}.pth', model)
    model.to(device)
    train_acc_rh = common.compute_acc_seq2seq(args, train_rh, model, device, writer, logging)
    val_acc_rh = common.compute_acc_seq2seq(args, val_rh, model, device, writer, logging)
    test_acc_rh = common.compute_acc_seq2seq(args, test_rh, model, device, writer, logging)

    model, _, _, _ = load_model(f'models/best_lh_{os.path.basename("#".join([x for x in args.values()]))}.pth', model)
    model.to(device)
    train_acc_lh = common.compute_acc_seq2seq(args, train_lh, model, device, writer, logging)
    val_acc_lh = common.compute_acc_seq2seq(args, val_lh, model, device, writer, logging)
    test_acc_lh = common.compute_acc_seq2seq(args, test_lh, model, device, writer, logging)

    print(f"Train: rh:{train_acc_rh:2.2%} lh:{train_acc_lh:2.2%}")
    print(f"Validation: rh:{val_acc_rh:2.2%} lh:{val_acc_lh:2.2%}")
    print(f"Test: rh:{test_acc_rh:2.2%} lh:{test_acc_lh:2.2%}")


if __name__ == '__main__':
    pass
