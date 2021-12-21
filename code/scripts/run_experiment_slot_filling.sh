#!/bin/bash

# Runs one experiment

echo "Current fold if given: $1"

cd ../source

source activate transformers  # activate your conda environment here

# For slot filling SciBERT model: semantics of parameters see main.py
python3 -u main.py -save_dir ../../models/scibertSlotFilling100epochs -subsampling 0.0 -batch_size 10 -task slot_typing \
    -optim adamW -epochs 100 -lr 1e-5 -lr_bert 1e-5 -adam_epsilon 1e-8 -weight_decay 0 -num_cross_val_folds 5 \
    -current_cross_val_fold $1 -model_type "BERT" -use_cuda -embeddings bert
