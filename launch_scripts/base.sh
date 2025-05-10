#!/bin/bash

# ===============================
# Dynamic base.sh
# Usage:
# bash base.sh baseline
# bash base.sh lr
# bash base.sh batchsize
# ===============================


EXP_NAME=$1

# Fallback if no name is passed
if [ -z "$EXP_NAME" ]; then
  echo "⚠️  Please provide an experiment name (e.g. baseline, lr, batchsize)"
  exit 1
fi

# Dynamic save path
SAVE_PATH="/kaggle/working/checkpoints_${EXP_NAME}"
DATA_PATH="/kaggle/working/datasets"

echo "🔧 Running experiment: $EXP_NAME"
echo "📁 Checkpoints will be saved to: $SAVE_PATH"

# 1) Run fine-tuning
python finetune.py \
  --data-location=$DATA_PATH \
  --exp_name $EXP_NAME

# 2) Evaluate single-task models
python eval_single_task.py \
  --data-location=$DATA_PATH \
  --save=$SAVE_PATH

# 3) Run task addition
python eval_task_addition.py \
  --data-location=$DATA_PATH \
  --save=$SAVE_PATH
