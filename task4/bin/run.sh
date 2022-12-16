#!/bin/bash

SHELL_FOLDER=$(dirname "$0")
cd $SHELL_FOLDER
PYTHONPATH=$(dirname $(dirname $(pwd)}))
export PYTHONPATH=$PYTHONPATH
cd ".."
echo "当前工作路径: $(pwd)"

for encoder_model in 'roberta-large' 
do
    for max_epochs in 35
    do
        for batch_size in 16
        do
            for model_type in 'ntr_focal_loss_argument_model' 'db_no_focal_loss_argument_model' 'class_balanced_ntr_loss_argument_model' \
                              'distribution_balanced_loss_argument_model' 'rbce_focal_loss_argument_model' 'baseline_argument_model' \
                              'focal_loss_argument_model' 'class_balanced_loss_argument_model'
            do
                for dataset_type in 'premise_argument_dataset'
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
                        --monitor 'val_f1' \
                        --monitor 'val_Self-direction: action@f1' \
                        --monitor 'Self-direction: thought@f1' \
                        --monitor 'Stimulation@f1' \
                        --monitor 'Hedonism@f1' \
                        --monitor 'Achievement@1' \
                        --monitor 'Power: dominance@f1' \
                        --monitor 'Power: resources@f1' \
                        --monitor 'Face@f1' \
                        --monitor 'Security: personal@f1' \
                        --monitor 'Security: societal@f1' \
                        --monitor 'Tradition@f1' \
                        --monitor 'Conformity: rules@f1' \
                        --monitor 'Conformity: interpersonal@f1' \
                        --monitor 'Humility@f1' \
                        --monitor 'Benevolence: caring@f1' \
                        --monitor 'Benevolence: dependability@f1' \
                        --monitor 'Universalism: concern@f1' \
                        --monitor 'Universalism: nature@f1' \
                        --monitor 'Universalism: tolerance@f1' \
                        --monitor 'Universalism: objectivity@f1'

                    done
                done
            done            
        done
    done 
done
