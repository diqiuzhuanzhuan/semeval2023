# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import argparse
from allennlp.common.params import Params
import pytorch_lightning as pl
from task4.data_man.argument_reader import ArgumentDataModule
from task4.modeling.model import ArgumentModel

def parse_arguments():

    parser = argparse.ArgumentParser(description='run experiment')

    parser.add_argument('--data_module_type', type=str, default='baseline_argument_data_module',help='')
    parser.add_argument('--dataset_type', type=str, default='baseline_argument_dataset', help='')
    parser.add_argument('--model_type', type=str, default='baseline_argument_model', help='')
    parser.add_argument('--encoder_model', type=str, default='bert-base-uncased', help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    
    args = parser.parse_args()

    return args 


if __name__ == '__main__':
    args = parse_arguments()
    trainer = pl.Trainer(max_epochs=1)
    adm = ArgumentDataModule.from_params(Params({
        'type': args.data_module_type,
        'reader': Params({
            'type': args.dataset_type
        }),
        'batch_size': args.batch_size
    }))

    params = Params({
        'type': args.model_type,
        'encoder_model': args.encoder_model
    })
    argument_model = ArgumentModel.from_params(params=params)
    trainer.fit(model=argument_model, datamodule=adm)

    