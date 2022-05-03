import random

import numpy as np

from loader import reverse_hand, load_binary
from visualization import visualize


def hand_symmetry(verbose=True):
    data = random.choice(load_binary('../data/musescore_fingers.pickle')['left']['train'])
    data = [np.array(data['pitchs']) / 127], [np.array(data['onsets'])], [0], [np.array(data['fingers'])], [0], [0], [0]
    if verbose:
        visualize(data[0][0], data[1][0], data[3][0])
    new_data = reverse_hand(data)
    if verbose:
        visualize(new_data[0][0], new_data[1][0], new_data[3][0])


if __name__ == '__main__':
    # for _ in range(1000):
    #     hand_symmetry(verbose=False)
    hand_symmetry(verbose=True)
