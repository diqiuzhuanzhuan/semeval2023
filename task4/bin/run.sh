#!/bin/bash

SHELL_FOLDER=$(dirname "$0")
cd $SHELL_FOLDER
PYTHONPATH=$(dirname $(dirname $(pwd)}))
export PYTHONPATH=$PYTHONPATH
cd ".."
echo "当前工作路径: $(pwd)"

for encoder_model in 'roberta-large' 'bert-base-uncased' 'bert-large-uncased'
do
    for max_epochs in 35
    do
        for batch_size in 16
        do
            for model_type in 'ntr_focal_loss_argument_model' 'db_no_focal_loss_argument_model' 'class_balanced_ntr_loss_argument_model' \
                              'distribution_balanced_loss_argument_model' 'rbce_focal_loss_argument_model' 'baseline_argument_model' \
                              'focal_loss_argument_model' 'class_balanced_loss_argument_model'
            do
                for dataset_type in 'rewrite_argument_dataset'
                do
                    for data_module_type in 'baseline_argument_data_module'
                    do
                        python apps/main.py \
                        --encoder_model $encoder_model \
                        --max_epochs $max_epochs \
                        --batch_size $batch_size \
                        --model_type $model_type \
                        --dataset_type $dataset_type \
                        --data_module_type $data_module_type \
                        --gpus -1 \
                        --cross_validation 1 \
                        --monitor 'val_f1' 

                    done
                done
            done            
        done
    done 
done
