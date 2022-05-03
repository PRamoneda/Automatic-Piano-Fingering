import torch

from common_classification import load_model
from nns import seq2seq_model, common, classification_model, windowed_model, f_seq2seq
from nns.nakamura_baselines import build_model


def choice_model(architecture, architecture_type):
    if architecture_type == 'windowed':
        if architecture == 'only_pitch:fc':
            model = windowed_model.windowed(
                embedding=common.only_pitch(),
                encoder=windowed_model.nakamura_encoder(1),
                decoder=common.without_decoder()
            )

    elif 'finetuning_windows' in architecture_type:
        arq = architecture.split(';')

        finetuning_type = arq[0]
        from_architecture = arq[1]

        if from_architecture == 'soft(only_pitch:lstm(dropout2))':
            model = classification_model.classification(
                embedding=common.only_pitch(),
                encoder=classification_model.lstm_encoder(input=1, dropout=.4),
            )
            model, _, _, _ = load_model(
                path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:lstm(dropout2)).pth',
                model=model
            )
        elif from_architecture == 'soft(only_pitch:fc)':
            model = classification_model.classification(
                embedding=common.only_pitch(),
                encoder=classification_model.fc_encoder(1),
            )
            model, _, _, _ = load_model(
                path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:fc).pth',
                model=model
            )
        elif from_architecture == 'soft(only_pitch:gnn)':
            model = classification_model.classification(
                embedding=common.only_pitch(),
                encoder=classification_model.gnn_encoder(input_size=1),
            )
            model, _, _, _ = load_model(
                path='models/base/best_augmented_noisy#augmented_noisy_w11#classification_augmented_noisy#soft(only_pitch:gnn).pth',
                model=model
            )

        if finetuning_type == 'last_layer':
            model.freeze_all()
            model.unfreeze_last_layer()

    elif 'finetuning_seq2seq' == architecture_type or 'finetuning_separated' == architecture_type or 'generalization' == architecture_type:
        """
            (GNN/LSTM)_(0/1/2/3)_(fc/gnn/lstm/ar4)
        """
        # encoder_type, freeze_layers, decoder = architecture.split('_')
        #
        # model = f_seq2seq.seq2seq(
        #     encoder_type=encoder_type,
        #     freeze_layers=freeze_layers,
        #     decoder=decoder
        # )
        model_alias, transfer_type = architecture.split('_')

        if model_alias == 'soanoisy':
            path_model = "models/base_seq2seq/soa_noisy.pth"
            model = seq2seq_model.seq2seq(
                embedding=common.only_pitch(),
                encoder=seq2seq_model.lstm_encoder(input=1, dropout=0.4),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == 'soa-gnn':
            path_model = 'models/base_seq2seq/gnn:ar.pth'
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == 'soa-soft-gnn':
            path_model = 'models/base_seq2seq/soft(gnn:ar).pth'
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == 'gnn:ar':
            path_model = 'models/best_official_base#augmented_noisy_random_seq2seq#seq2seq_noisy#gnn:ar.pth'
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == 'lstm:ar':
            path_model = 'models/best_official_base#augmented_noisy_random_seq2seq#seq2seq_noisy#lstm:ar.pth'
            model = seq2seq_model.seq2seq(
                embedding=common.only_pitch(),
                encoder=seq2seq_model.lstm_encoder(input=1, dropout=0.0),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == 'noaugsoft(gnn:ar)':
            path_model = 'models/best_ablation_base_augmentation#no_augmented_noisy_random_seq2seq#seq2seq_noisy#soft(gnn:ar).pth'
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == 'noauglstm:ar':
            path_model = 'models/best_ablation_base_augmentation#no_augmented_noisy_random_seq2seq#seq2seq_noisy#lstm:ar.pth'
            model = seq2seq_model.seq2seq(
                embedding=common.only_pitch(),
                encoder=seq2seq_model.lstm_encoder(input=1, dropout=0.0),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == 'soft(gnn:ar)':
            path_model = 'models/best_official_base#augmented_noisy_random_seq2seq#seq2seq_noisy#soft(gnn:ar).pth'
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == 'soft(lstm:ar)':
            path_model = 'models/best_official_base#augmented_noisy_random_seq2seq#seq2seq_noisy#soft(lstm:ar).pth'
            model = seq2seq_model.seq2seq(
                embedding=common.only_pitch(),
                encoder=seq2seq_model.lstm_encoder(input=1, dropout=0.0),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif model_alias == "lstm:fc":
            path_model = 'models/best_ablation_base_autoregressive#augmented_noisy_random_seq2seq#seq2seq_noisy#lstm:fc.pth'
            model = seq2seq_model.seq2seq(
                embedding=common.only_pitch(),
                encoder=seq2seq_model.lstm_encoder(input=1, dropout=0),
                decoder=seq2seq_model.linear_decoder()
            )
        elif model_alias == "soft(gnn:fc)":
            path_model = 'models/best_ablation_base_autoregressive#augmented_noisy_random_seq2seq#seq2seq_noisy#soft(gnn:fc).pth'
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.linear_decoder()
            )
        elif model_alias == "soft(gnn:fc)":
            path_model = 'models/best_ablation_base_autoregressive#augmented_noisy_random_seq2seq#seq2seq_noisy#soft(gnn:fc).pth'
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.linear_decoder()
            )
        else:
            print(f"ERROR: {architecture} {model_alias}")

        if transfer_type != '-1':
            print(f'path_model is {path_model}')
            model, _, _, _ = load_model(
                path=path_model,
                model=model,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            )
            model.freeze(transfer_type)
    elif architecture_type in ['classification', 'classification_noisy', 'classification_augmented', 'classification_augmented_noisy']:
        if architecture in ['only_pitch:gnn', 'soft(only_pitch:gnn)']:
            model = classification_model.classification(
                embedding=common.only_pitch(),
                encoder=classification_model.gnn_encoder(input_size=1),
            )
        elif architecture == 'nakamura_baseline_lstm':
            model = build_model(model_name='LSTM', input_lengths=11, hidden_size=32, num_class=5)
        elif architecture == 'nakamura_baseline_ff':
            model = build_model(model_name='Forward', input_lengths=11, hidden_size=32, num_class=5)
        elif architecture in ['only_pitch:fc', 'l2_regularization(only_pitch:fc)', 'soft(only_pitch:fc)']:
            model = classification_model.classification(
                embedding=common.only_pitch(),
                encoder=classification_model.fc_encoder(1),
            )
        elif architecture == 'only_pitch:gru':
            model = classification_model.classification(
                embedding=common.only_pitch(),
                encoder=classification_model.gru_encoder(input=1),
            )
        elif architecture in ['only_pitch:lstm', 'l2_regularization(only_pitch:lstm)', 'soft(only_pitch:lstm)']:
            model = classification_model.classification(
                embedding=common.only_pitch(),
                encoder=classification_model.lstm_encoder(input=1),
            )
        elif architecture == 'soft(only_pitch:lstm(dropout2))':
            model = classification_model.classification(
                embedding=common.only_pitch(),
                encoder=classification_model.lstm_encoder(input=1, dropout=.4),
            )
        elif architecture in ['simple:gnn', 'soft(simple:gnn)']:
            model = classification_model.classification(
                embedding=classification_model.embeddingSimple(),
                encoder=classification_model.gnn_encoder(input_size=3),
            )
        elif architecture == 'simple:fc':
            model = classification_model.classification(
                embedding=classification_model.embeddingSimple(),
                encoder=classification_model.fc_encoder(10),
            )
        elif architecture == 'simple:gru':
            model = classification_model.classification(
                embedding=classification_model.embeddingSimple(),
                encoder=classification_model.gru_encoder(input=10),
            )
        elif architecture == 'simple:lstm':
            model = classification_model.classification(
                embedding=classification_model.embeddingSimple(),
                encoder=classification_model.lstm_encoder(input=10),
            )
        elif architecture == 'vicent:gnn':
            model = classification_model.classification(
                embedding=classification_model.embeddingVicent(),
                encoder=classification_model.gnn_encoder(input_size=5),
            )
        elif architecture == 'vicent:fc':
            model = classification_model.classification(
                embedding=classification_model.embeddingVicent(),
                encoder=classification_model.fc_encoder(input=64),
            )
        elif architecture == 'vicent:gru':
            model = classification_model.classification(
                embedding=classification_model.embeddingVicent(),
                encoder=classification_model.gru_encoder(input=64),
            )
        elif architecture == 'vicent:lstm':
            model = classification_model.classification(
                embedding=classification_model.embeddingVicent(),
                encoder=classification_model.lstm_encoder(input=64),
            )
        elif architecture in ['without:gnn', 'soft(without:gnn)']:
            model = classification_model.classification(
                embedding=common.without_embedding(),
                encoder=classification_model.gnn_encoder(input_size=3),
            )
        elif architecture in ['without:fc', 'l2_regularization(without:fc)', 'soft(without:fc)']:
            model = classification_model.classification(
                embedding=common.without_embedding(),
                encoder=classification_model.fc_encoder(input=3),
            )
        elif architecture == 'without:gru':
            model = classification_model.classification(
                embedding=common.without_embedding(),
                encoder=classification_model.gru_encoder(input=3),
            )
        elif architecture in ['without:lstm', 'l2_regularization(without:lstm)', 'soft(without:lstm)']:
            model = classification_model.classification(
                embedding=common.without_embedding(),
                encoder=classification_model.lstm_encoder(input=3),
            )
        elif architecture == 'soft(without:lstm(dropout2))':
            model = classification_model.classification(
                embedding=common.without_embedding(),
                encoder=classification_model.lstm_encoder(input=3, dropout=.4),
            )
        else:
            print(f"ERROR: {architecture}")

    elif architecture_type in ['seq2seq', 'seq2seq_noisy']:
        if architecture == "lstm:ar" or architecture == "soft(lstm:ar)":
            model = seq2seq_model.seq2seq(
                embedding=common.only_pitch(),
                encoder=seq2seq_model.lstm_encoder(input=1, dropout=0),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif architecture == "lstm:fc" or architecture == "soft(lstm:fc)]":
            model = seq2seq_model.seq2seq(
                embedding=common.only_pitch(),
                encoder=seq2seq_model.lstm_encoder(input=1, dropout=0),
                decoder=seq2seq_model.linear_decoder()
            )
        elif architecture == "gnn:fc" or architecture == "soft(gnn:fc)":
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.linear_decoder()
            )
        elif architecture == "gnn:ar" or architecture == "soft(gnn:ar)":
            model = seq2seq_model.seq2seq(
                embedding=common.emb_pitch(),
                encoder=seq2seq_model.gnn_encoder(input_size=64),
                decoder=seq2seq_model.AR_decoder(64)
            )
        elif architecture == "lstm_dropout:ar" or architecture == "soft(lstm_dropout:ar)" or architecture == "weight_decay_soft(lstm_dropout:ar)":
            model = seq2seq_model.seq2seq(
                embedding=common.only_pitch(),
                encoder=seq2seq_model.lstm_encoder(input=1, dropout=0.4),
                decoder=seq2seq_model.AR_decoder(64)
            )
        else:
            print(f"ERROR: {architecture}")

    return model