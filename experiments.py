import os
import sys

import torch

import experiment_generalization
import finetuning_seq2seq
import finetuning_seq2seq_separated
import finetuning_windows
import training
import training_augmented
import training_augmented_noisy
import training_augmented_noisy_seq2seq
from choice_model import choice_model
from legacy import training_windowed, finetuning_seq2seq_fair
import training_noisy
from datetime import datetime

if __name__ == '__main__':
    start = datetime.now()
    # os.nice(-15)
    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists('models'):
        os.mkdir('models')

    alias, rep, architecture_type, architecture = sys.argv[1].split('#')
    args = {
        'alias': alias,
        'rep': rep,
        'architecture_type': architecture_type,
        'architecture': architecture
    }

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    model = choice_model(architecture, architecture_type)

    if architecture_type == 'windowed':
        tr = training_windowed
    elif architecture_type == 'classification':
        tr = training
    elif architecture_type == 'classification_noisy':
        tr = training_noisy
    elif architecture_type == 'classification_augmented':
        tr = training_augmented
    elif architecture_type == 'classification_augmented_noisy':
        tr = training_augmented_noisy
    elif architecture_type == 'generalization':
        tr = experiment_generalization
    elif architecture_type == 'seq2seq_noisy':
        tr = training_augmented_noisy_seq2seq
    elif 'finetuning_windows' in architecture_type:
        tr = finetuning_windows
    elif 'finetuning_seq2seq' in architecture_type:
        tr = finetuning_seq2seq
    elif 'finetuning_separated' in architecture_type:
        tr = finetuning_seq2seq_separated

    data = tr.create_dataset(rep)
    writer, logging = tr.training_loop(data, device, model, args)
    tr.run_test(data, device, model, args, writer)

    if logging:
        writer.add_text("time cost", str(datetime.now()-start))
