# Automatic piano fingering from partially annotated scores using autoregressive neural networks

Link to the paper will be here (currently, the paper is under peer review).

## Abstract

Piano fingering is a creative and highly individualised task acquired by musicians progressively in their first music education years. Pianists must learn to choose the order of fingers to play the piano keys because scores do not have engraved finger and hand movements as other technique elements. Numerous research efforts have been conducted for automatic piano fingering based on a previous dataset composed of 150 score excerpts fully annotated by multiple expert annotators. However, most piano sheets include partial annotations for problematic finger and hand movements. We introduce a novel dataset for the task, the ThumbSet dataset, containing 2523 pieces with partial and noisy annotations of piano fingering crowdsourced from non-expert annotators. As part of our methodology, we propose two autoregressive neural networks with beam search decoding for modelling automatic piano fingering as a sequence-to-sequence learning problem, considering the correlation between output finger labels. We design the first model with the exact pitch representation of previous proposals. The second model uses graph neural networks to more effectively represent polyphony, whose treatment has been a common issue across previous studies. Finally, we finetune the models on the existing expert annotations dataset. The evaluation shows that (1) we are able to achieve high performance when training on the ThumbSet dataset and that (2) the proposed models outperform the state-of-the-art hidden Markov models and recurrent neural network baselines. Code, dataset, models, and results are made available to enhance the task reproducibility, including a new framework for evaluation

## Project Structure

- `finetuning_seq2seq_separated.py`: finetuning over PIG dataset.

- `training_augmented_noisy_seq2seq.py`: train on Thumbset domain with partial annotations

- `loader.py`: creates the different representations.

- Directory `results` contains the results.


- Directory `data` contains the processed data.

- Directory `nns` contains the pytorch models.

- Directories `PianoFingeringDataset_v1.02` and `ThumbSet_v1.01` should be filled by both "upon request" automatic piano fingering datasets available at: (https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/) and (https://zenodo.org/record/6433702#.YnFFYvNBxhE).

## Train

To re-train `ArLSTM` and `ArGNN` execute: `bash train.bash`.

## Pre-trained models

The models presented on the paper, `ArLSTM` and `ArGNN`, are available in the directory `models`.

## Cite

blind review
