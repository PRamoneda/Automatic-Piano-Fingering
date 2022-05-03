import pdb

import torch

from common_classification import load_model, compute_acc_seq2seq
from nns import seq2seq_model, common
from training_augmented_noisy_seq2seq import create_dataset

# freezing
# 100%|███████████████████████████████████| 178731/178731 [51:45<00:00, 57.56it/s]
# base noisy model acc in noisy data 0.5694918101802542
# 100%|███████████████████████████████████| 178731/178731 [51:50<00:00, 57.47it/s]
# base rh model acc in noisy data 0.5699255408592178

# f = open("demofile.txt", "r")
# print(f.readlines())

test_rh,\
test_lh,\
noisy_validation_rh,\
noisy_validation_lh,\
noisy_windowed = create_dataset('augmented_noisy_random_seq2seq', batch_size_training=1)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

one_batch = next(iter(noisy_windowed))
# pdb.set_trace()

model = seq2seq_model.seq2seq(
    embedding=common.only_pitch(),
    encoder=seq2seq_model.lstm_encoder(input=1, dropout=0),
    decoder=seq2seq_model.AR_decoder(64)
)



path_model = 'experiment_cf#nakamura_augmented_seq2seq_merged#finetuning_seq2seq#soanoisy_0.pth'
path_noisy_model = 'models/base_seq2seq/soa_noisy.pth'

model_noisy, _, _, _ = load_model(
    path=path_noisy_model,
    model=model,
    device=device
)

model_rh, _, _, _ = load_model(
    path=f"models/best_rh_{path_model}",
    model=model,
    device=device
)

model_lh, _, _, _ = load_model(
    path=f"models/best_lh_{path_model}",
    model=model,
    device=device
)



model_noisy.to(device)
print(f"base noisy model acc in noisy data {compute_acc_seq2seq(noisy_windowed, device, model=model_noisy)}")
model_rh.to(device)
print(f"base rh model acc in noisy data {compute_acc_seq2seq(noisy_windowed, device, model=model_rh)}")
model_lh.to(device)
print(f"base lh model acc in noisy data {compute_acc_seq2seq(noisy_windowed, device, model=model_lh)}")