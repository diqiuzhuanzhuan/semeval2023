#!/bin/bash

SHELL_FOLDER=$(dirname "$0")
cd $SHELL_FOLDER
PYTHONPATH=$(dirname $(dirname $(pwd)}))
export PYTHONPATH=$PYTHONPATH
cd ".."
echo "当前工作路径: $(pwd)"
export PYTHONPATH="$PWD/..":$PYTHONPATH

for encoder_model in  'bert-base-uncased' 'bert-large-uncased'
do
    for max_epochs in 20
    do
        for batch_size in 16
        do
            for model_type in 'baseline_argument_model'
            do
                for dataset_type in 'baseline_argument_dataset'
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
                        --gpus 1
                    done
                done
            done            
        done
    done 
done
