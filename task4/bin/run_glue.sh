#!/bin/bash

SHELL_FOLDER=`realpath $(dirname "$0")`
echo "shell folder: $SHELL_FOLDER"
TASK4_FOLDER=$(dirname $SHELL_FOLDER)
echo "task4 folder: $TASK4_FOLDER"
PROJECT_DIR=$(dirname $TASK4_FOLDER)
echo "project_dir: $PROJECT_DIR"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

python $PROJECT_DIR/task4/apps/run_glue.py \
  --model_name_or_path 'bert-base-uncased' \
  --dataset_name $PROJECT_DIR/task4/data_man/semeval_hf_dataset.py \
  --dataset_config_name original \
  --output_dir $PROJECT_DIR/task4/output \
  --overwrite_output_dir \
  --logging_dir $PROJECT_DIR/task4/log \
  --logging_strategy epoch \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --warmup_steps 200 \
  --metric_for_best_model macro_f1 \
  --greater_is_better \
  --load_best_model_at_end \
  --do_train \
  --do_eval \
  --fp16 \
  --fp16_opt_level O2 \
  --num_train_epochs 30 \
  --save_total_limit 3 \
  --seed 42
