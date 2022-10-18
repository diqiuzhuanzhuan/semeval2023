# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import argparse
import os
import time
from typing import Union
from allennlp.common.params import Params
import pytorch_lightning as pl
from task4.data_man.argument_reader import ArgumentDataModule
from task4.modeling.model import ArgumentModel
from task4.configuration.config import logging
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def parse_arguments():

    parser = argparse.ArgumentParser(description='run experiment')

    parser.add_argument('--data_module_type', type=str, default='baseline_argument_data_module',help='')
    parser.add_argument('--dataset_type', type=str, default='baseline_argument_dataset', help='')
    parser.add_argument('--model_type', type=str, default='baseline_argument_model', help='')
    parser.add_argument('--encoder_model', type=str, default='bert-base-uncased', help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--max_epochs', type=int, default=2, help='')
    
    args = parser.parse_args()

    return args 


def get_model_earlystopping_callback(monitor='val_f1', mode:Union['max', 'min']='max', min_delta=0.001):
    if "f1" in monitor.lower():
        
        es_clb = EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=3,
            verbose=True,
            mode=mode
        )
    else:
        es_clb = EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=3,
            verbose=True,
            mode=mode
        )
    return es_clb


def get_model_best_checkpoint_callback(dirpath='checkpoints', monitor='val_f1', mode:Union['max', 'min']='max'):
    bc_clb = ModelCheckpoint(
        filename='{{epoch}}-{{{}:.3f}}-{{val_loss:.2f}}'.format(monitor),
        save_top_k=1,
        verbose=True,
        monitor=monitor,
        mode=mode
        )
    return  bc_clb


def save_model(trainer: pl.Trainer, default_root_dir="", model_name='', timestamp=None):
    out_dir = default_root_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)

    logging.info('Stored model {}.'.format(outfile))
    best_checkpoint = None
    for file in os.listdir(out_dir):
        if file.startswith("epoch"):
            best_checkpoint = os.path.join(out_dir, file)
            break
    return outfile, best_checkpoint

def load_model(model_class: pl.LightningModule, model_file, stage='test', **kwargs):
    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    model = model_class.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage)
    model.stage = stage
    return model
    

if __name__ == '__main__':
    args = parse_arguments()
    callbacks = [get_model_earlystopping_callback(), get_model_best_checkpoint_callback()]
    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=callbacks)
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

    