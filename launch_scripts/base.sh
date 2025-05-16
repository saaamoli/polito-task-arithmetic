#!/bin/bash

# ===============================
# Portable base.sh
# Usage:
# bash base.sh baseline /path/to/project/root
# ===============================

EXP_NAME=$1
PROJECT_ROOT=$2

# Check if experiment name is provided
if [ -z "$EXP_NAME" ]; then
  echo "⚠️  Please provide an experiment name (e.g. baseline, lr, batchsize)"
  exit 1
fi

# Default project root to current directory if not provided
if [ -z "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="."
fi

SAVE_PATH="${PROJECT_ROOT}/checkpoints_${EXP_NAME}"
DATA_PATH="${PROJECT_ROOT}/datasets"

echo "🔧 Running experiment: $EXP_NAME"
echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Checkpoints will be saved to: $SAVE_PATH"
echo "📂 Datasets should be in: $DATA_PATH"

# 1) Run fine-tuning
python finetune.py \
  --data-location=$PROJECT_ROOT \
  --exp_name $EXP_NAME

# 2) Evaluate single-task models
python eval_single_task.py \
  --data-location=$PROJECT_ROOT \
  --exp_name $EXP_NAME

# 3) Run task addition
python eval_task_addition.py \
  --data-location=$PROJECT_ROOT \
  --exp_name $EXP_NAME
