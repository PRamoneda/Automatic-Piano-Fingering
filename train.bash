#!/bin/bash

run(){
    echo $1
    python3 experiments.py $1 | tee output/$1.txt
}

python3 loader.py

run "official_base#augmented_noisy_random_seq2seq#seq2seq_noisy#soft(lstm:ar)"
run "official_base#augmented_noisy_random_seq2seq#seq2seq_noisy#gnn:ar"

run "official#nakamura_augmented_seq2seq_separated#finetuning_separated#soft(gnn:ar)_0"
run "official#nakamura_augmented_seq2seq_separated#finetuning_separated#lstm:ar_1"