from random import choice

import numpy as np

from loader import musescore_dataset, load_binary, normalize_midi, normalize_data, compute_edge_list_window
from utils import save_json
from visualization import visualize


def musescore_dataset(set_name, hand='right', w_type='w11'):
    if w_type == 'w11':
        windows = load_binary('data/musescore_fingers.pickle')[hand][set_name]
    elif w_type == 'random':
        windows = load_binary('data/musescore_fingers_w_up64.pickle')[hand][set_name]

    w = choice(windows)
    note = normalize_midi(np.array(w['pitchs'], dtype=float))
    onset = normalize_data(np.array(w['onsets'], dtype=float))
    duration = normalize_data(np.array(w['offsets'], dtype=float) - np.array(w['onsets'], dtype=float))
    finger = np.array(w['fingers'])
    ids = (w['alias'].split('.')[0], w['alias'][-1])
    lengths = len(w['pitchs'])
    edges = compute_edge_list_window(w)
    return note, onset, duration, finger, ids, lengths, edges


if __name__ == '__main__':
    edges = {}
    for ii in range(0, 10):
        data = musescore_dataset('train', hand='right', w_type='random')
        for dd in data:
            print(dd)
        visualize(data[0], data[1], data[3], save=f'/Users/pedro/Downloads/{ii}.musicxml')
        edges[ii] = [list(dd) for dd in data[6]]


    save_json(edges, '/Users/pedro/Downloads/edges.json')
