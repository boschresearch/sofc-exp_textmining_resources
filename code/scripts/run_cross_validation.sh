#!/bin/bash

# Runs 5-fold cross validation for SOFC-Exp corpus related experiments.
# Paths assume that this script is run from within the scripts folder.

for f in {1..5}; do

# Uncomment one of the run_experiment... lines depending on which experiments you want to do.
# Starts five runs in parallel - change if your computing environment allows only one GPU process at a time,
# add call to scheduler, etc.

# Variations of the models tested are configured in the respective run_experiments... file.

  # Experiment
  # run_experiment_sentence_classification.sh $f

  # Entity Types
  # run_experiment_entity_typing.sh $f

  # Slot filling
  # run_experiment_slot_filling.sh $f
done

# The above processes all write their results to prediction files.
# Once they are done, collect results and compute performance statistics.
# Use the file source/evaluation/evaluate_cross_validation.py with appropriate command line arguments.

# Examples: 
# sentence classification:
# python -u evaluation/evaluate_cross_validation.py -predictions_dir DIRECTORY_WITH_MODEL_PREDICTIONS -eval_mode "multiclass" -num_labels 2
# entity typing:
# python -u evaluation/evaluate_cross_validation.py -predictions_dir DIRECTORY_WITH_MODEL_PREDICTIONS -eval_mode "conll" -task "entity_types"
# slot typing:
# python -u evaluation/evaluate_cross_validation.py -predictions_dir DIRECTORY_WITH_MODEL_PREDICTIONS -eval_mode "conll" -task "slot_types"
