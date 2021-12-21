#!/bin/bash

# Runs one experiment

echo "Current fold if given: $1"

cd ../source

source activate transformers  # activate your conda envinroment here

# For sentence classification BERT model: semantics of parameters see main.py
python3 -u main.py -save_dir ../../models/bertLarge100epochs -subsampling 0.3 -batch_size 10 -task sentence \
    -optim adamW -epochs 100 -lr 4e-7 -adam_epsilon 1e-8 -weight_decay 0 -num_cross_val_folds 5 \
    -current_cross_val_fold $1 -model_type "BERT" -use_cuda -embeddings bert

