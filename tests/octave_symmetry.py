import random

import numpy as np

from loader import load_binary, octave_symmetry, normalize_data
from visualization import visualize


def test_octave_symmetry():
    data = random.choice(load_binary('../data/musescore_fingers.pickle')['left']['train'])
    notes, onsets, durations, fingers, idx, lengths, edges = np.array(data['pitchs']) / 127, normalize_data(np.array(data['onsets'])), [], data['fingers'], [], [], []
    visualize(notes, onsets, fingers)
    new_data = octave_symmetry(notes, onsets, durations, fingers, idx, lengths, edges)
    for notes, onsets, durations, fingers, idx, lengths, edges in zip(*new_data):
        visualize(notes, onsets, fingers)


if __name__ == '__main__':
    test_octave_symmetry()