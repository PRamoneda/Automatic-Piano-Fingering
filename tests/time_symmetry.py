import random

import numpy as np

from loader import load_binary, time_symmetry, normalize_data
from visualization import visualize


def test_time_symmetry():
    data = random.choice(load_binary('../data/musescore_fingers.pickle')['left']['train'])
    notes, onsets, durations, fingers, idx, lengths, edges = np.array(data['pitchs']) / 127, normalize_data(np.array(data['onsets'])), [], data['fingers'], [], [], []
    visualize(notes, onsets, fingers)
    new_data = time_symmetry(notes, onsets, durations, fingers, idx, lengths, edges)
    visualize(new_data[0], new_data[1], new_data[3])


if __name__ == '__main__':
    test_time_symmetry()
