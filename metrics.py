import os
import pdb

import torch

from loader import filter_edges, salami_tensor, load_note2ids
from legacy.training_windowed import compute_edges_batch
from utils import save_json, load_json


def average_annotator_per_piece(ids, match_rates):
    # compute average_annotator_per_piece
    number, _ = zip(*ids)
    mmr_all = []
    number = list(set(number))
    for n in number:
        mrs = [mr for id_piece, mr in zip(number, match_rates) if id_piece == n]
        mmr = sum(mrs) / len(mrs)
        mmr_all.append(mmr)
    # averaging all the multi-annotator match rates
    return mmr_all, number


def general_match_rate(y_pred, y_true, ids, lengths=0):
    # indicating how closely the estimation agrees with all the ground truths
    # compute match rate for every piece fingered

    if lengths == 0:
        lengths = [len(yy) for yy in y_pred]

    match_rates = []
    for p, t, l, id_piece in zip(y_pred, y_true, lengths, ids):
        assert len(p) == len(t) == l, f"id {id_piece}: apples with lemons gmr: {len(p)} != {len(t)} != {l}"
        matches = 0
        for idx, (pp, tt) in enumerate(zip(p, t)):
            if idx >= l:
                break
            else:
                if pp == tt:
                    matches += 1
        match_rates.append(matches/l)
    return average_annotator_per_piece(ids, match_rates)


def avg_general_match_rate(y_pred, y_true, ids, lengths=0):
    gmr, _ = general_match_rate(y_pred, y_true, ids, lengths=0)
    return sum(gmr) / len(gmr)


def highest_match_rate(y_pred, y_true, ids, lengths=0):
    # focus on the ground truth closest to the estimation
    if lengths == 0:
        lengths = [len(yy) for yy in y_pred]
    # pdb.set_trace()
    match_rates = []
    for p, t, l in zip(y_pred, y_true, lengths):
        matches = 0
        for idx, (pp, tt) in enumerate(zip(p, t)):
            if idx >= l:
                break
            else:
                if pp == tt:
                    matches += 1

        match_rates.append(matches / l)

    return average_annotator_per_piece(ids, match_rates)


def avg_highest_match_rate(y_pred, y_true, ids, lengths=0):
    hmr_all, _ = highest_match_rate(y_pred, y_true, ids, lengths=lengths)
    return sum(hmr_all) / len(hmr_all)


def create_dictionaries_ids(y_true, y_pred, lengths, ids, hand):
    IDS2NOTES = load_note2ids()
    id2fingers_true = []
    id2fingers_pred = []
    for t, p, l, (id_score, id_annotator) in zip(y_true, y_pred, lengths, ids):
        ids_PIG = IDS2NOTES['_'.join((id_score, id_annotator))][hand]
        id2pred = {}
        id2true = {}
        for idx, (id_PIG, tt, pp) in enumerate(zip(ids_PIG, t, p)):
            id2pred[id_PIG] = pp
            id2true[id_PIG] = tt
        id2fingers_true.append(id2true)
        id2fingers_pred.append(id2pred)
    return id2fingers_true, id2fingers_pred


def soft_match_rate(y_pred, y_true, ids, lengths=0, hand='right'):
    # the softest criterion of correct estimation for each note is to judge whether the estimated finger matches at
    # least one of the ground truths
    # focus on the ground truth closest to the estimation
    if lengths == 0:
        lengths = [len(yy) for yy in y_pred]

    # create dictionaries
    id2fingers_true, id2fingers_pred = create_dictionaries_ids(y_true, y_pred, lengths, ids, hand)

    # compute soft match rate from each piece and annotator
    id_pieces, id_annotators = zip(*ids)
    mr_all, number = [], []
    for n, a, id2finger_pred in zip(id_pieces, id_annotators, id2fingers_pred):
        matches_id2finger = [
            di2finger_dict for id_piece, id_annotator, di2finger_dict in zip(id_pieces, id_annotators, id2fingers_true)
            if id_piece == n
        ]
        number.append(n)
        mrs = []
        for id_pred, finger_pred in id2finger_pred.items():
            match_note = 0
            for id2finger_true in matches_id2finger:
                if id_pred in id2finger_true and id2finger_true[id_pred] == finger_pred:
                    match_note = 1
            mrs.append(match_note)
        mr_all.append(sum(mrs) / len(mrs))
    return average_annotator_per_piece(ids, mr_all)


def avg_soft_match_rate(y_pred, y_true, ids, lengths=0, hand='right'):
    smr_all, _ = soft_match_rate(y_pred, y_true, ids, lengths=0, hand=hand)
    return sum(smr_all) / len(smr_all)


def recombination_match_rate(y_pred, y_true, ids, lengths=0):
    # an edit cost metric
    pass



# def compute_results_windowed(args, loader_set, model, device):
#     model.eval()
#     preds = []
#     trues = []
#     total_lengths = []
#     total_ids = []
#     for notes, onsets, durations, fingers, ids, lengths, edge_list in loader_set:
#
#         fingers_piece = []
#         for n, o, d, e in zip(salami_tensor(notes, window_size=11, hop_size=1),
#                              salami_tensor(onsets, window_size=11, hop_size=1),
#                              salami_tensor(durations, window_size=11, hop_size=1),
#                              filter_edges(edge_list[0], window_size=11, hop_size=1, len_notes=lengths[0].tolist())):
#
#             e = compute_edges_batch([e], lengths)
#             n = n.to(device)
#             o = o.to(device)
#             d = d.to(device)
#             e = e.to(device)
#             pdb.set_trace()
#             out = model(n, o, d, torch.Tensor([11]).to(device), e)
#             fingers_piece.append(out.argmax(dim=2).cpu().tolist()[0][5])
#
#         preds.append(fingers_piece)
#         trues.append(fingers.tolist()[0])
#         total_lengths.append(lengths.cpu().tolist()[0])
#         total_ids.append(ids[0])
#     pdb.set_trace()
#     gmr = avg_general_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
#     hmr = avg_highest_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
#     smr = avg_soft_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
#
#     subset = 'train'
#     if loader_set.subset == 2:
#         subset = 'validation'
#     elif loader_set.subset == 3:
#         subset = 'test'
#     elif loader_set.subset == 5:
#         subset = 'train&validation'
#     elif loader_set.subset == 4:
#         subset = 'windowed'
#
#     json_path = f"results/{args['alias']}#{args['rep']}#{args['architecture_type']}.json"
#     if not os.path.exists(json_path):
#         save_json({}, json_path)
#     new_json = load_json(json_path)
#     if args['architecture'] not in new_json:
#         print(write_number_parameters(model))
#         new_json[args['architecture']] = {
#             'params': write_number_parameters(model),
#             'train': {},
#             'validation': {},
#             'test': {}
#         }
#     if 'gmr' not in new_json[args['architecture']][subset] or new_json[args['architecture']][subset]['gmr'] < gmr:
#         new_json[args['architecture']][subset] = {
#             'gmr': gmr,
#             'hmr': hmr,
#             'smr': smr
#         }
#     save_json(new_json, json_path)
#
#     return gmr


# def compute_results(args, loader_set, model, device):
#     model.eval()
#     preds = []
#     trues = []
#     total_lengths = []
#     total_ids = []
#     for notes, onsets, durations, fingers, ids, lengths, edge_list in loader_set:
#         # pdb.set_trace()
#         notes = notes.to(device)
#         onsets = onsets.to(device)
#         durations = durations.to(device)
#         lengths = lengths.to(device)
#         edge_list = edge_list.to(device)
#         out = model(notes, onsets, durations, lengths, edge_list)
#         preds.extend(out.argmax(dim=2).cpu().tolist())
#         trues.extend(fingers.tolist())
#         total_lengths.extend(lengths.cpu().tolist())
#         total_ids.extend(ids)
#
#     gmr = avg_general_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
#     hmr = avg_highest_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
#     smr = avg_soft_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
#
#     subset = 'train'
#     if loader_set.subset == 2:
#         subset = 'validation'
#     elif loader_set.subset == 3:
#         subset = 'test'
#     elif loader_set.subset == 5:
#         subset = 'train&validation'
#     elif loader_set.subset == 4:
#         subset = 'train'
#
#     json_path = f"results/{args['alias']}#{args['rep']}#{args['architecture_type']}.json"
#     if not os.path.exists(json_path):
#         save_json({}, json_path)
#     new_json = load_json(json_path)
#     if args['architecture'] not in new_json:
#         print(write_number_parameters(model))
#         new_json[args['architecture']] = {
#             'params': write_number_parameters(model),
#             'train': {},
#             'validation': {},
#             'test': {}
#         }
#     if 'gmr' not in new_json[args['architecture']][subset] or new_json[args['architecture']][subset]['gmr'] < gmr:
#         new_json[args['architecture']][subset] = {
#             'gmr': gmr,
#             'hmr': hmr,
#             'smr': smr
#         }
#     save_json(new_json, json_path)
#
#     return gmr


def write_number_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params
