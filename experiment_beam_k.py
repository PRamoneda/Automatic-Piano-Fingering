import torch

import common_classification
import training_augmented_noisy_seq2seq
from common_classification import load_model
from loader import load_dasaem_test
from nns import seq2seq_model, common
from visualization.show_score import compute_fingers

model = seq2seq_model.seq2seq(
    embedding=common.only_pitch(),
    encoder=seq2seq_model.lstm_encoder(input=1, dropout=0),
    decoder=seq2seq_model.AR_decoder(64)
)


# model = seq2seq_model.seq2seq(
#     embedding=common.emb_pitch(),
#     encoder=seq2seq_model.gnn_encoder(input_size=64),
#     decoder=seq2seq_model.AR_decoder(64)
# )
# best_lh_experiment_validation_finetuning#validation_experiment_seq2seq_2#finetuning_seq2seq#soanoisy_1.pth
path_model = 'separated#nakamura_augmented_seq2seq_separated#finetuning_separated#soft(lstm:ar)_1.pth'
model_lh, _, _, _ = load_model(
    path=f"models/best_lh_{path_model}",
    model=model,
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)

model_rh, _, _, _ = load_model(
    path=f"models/best_rh_{path_model}",
    model=model,
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)


test_rh, test_lh = load_dasaem_test()
test_rh_loader = common_classification.create_loader(test_rh, 5, num_workers=1, batch_size=1, collate_fn=training_augmented_noisy_seq2seq.collate_fn_seq2seq)
test_lh_loader = common_classification.create_loader(test_lh, 6, num_workers=1, batch_size=1, collate_fn=training_augmented_noisy_seq2seq.collate_fn_seq2seq)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
model.to(device)
for k in [1, 6, 12, 18, 24]:
    acc_rh = common_classification.compute_results_seq2seq_with_beam(None, test_rh_loader, model, device, None, False, beam_k=k)
    acc_lh = common_classification.compute_results_seq2seq_with_beam(None, test_lh_loader, model, device, None, False, beam_k=k)
    print(f"beamk={k} Test (General match rate): rh:{acc_rh:2.2%} lh:{acc_lh:2.2%}")

# for output the score
# compute_fingers(
#         model,
#         "001",  # 002, .... 010... 021.... 030
#         None,
#         hand='right'
# )
