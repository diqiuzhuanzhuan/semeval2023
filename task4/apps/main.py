# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import argparse
import collections
from sklearn.model_selection import KFold
from pathlib import Path
import pandas as pd
import os, re, sys
import time
from tqdm import tqdm
from typing import AnyStr, Dict, List, Union
from allennlp.common.params import Params
import pytorch_lightning as pl
from task4.data_man.argument_reader import ArgumentDataModule
from task4.modeling.model import ArgumentModel
from task4.configuration.config import logging
import torch
from task4.configuration import config
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from task4.data_man.badcases import calc_metric_by_file
from task4.data_man.meta_data import get_header_from_label_file
from task4.apps.ensemble import vote


def parse_arguments():

    parser = argparse.ArgumentParser(description='run experiments')

    parser.add_argument('--data_module_type', type=str, default='baseline_argument_data_module',help='')
    parser.add_argument('--dataset_type', type=str, default='baseline_argument_dataset', help='')
    parser.add_argument('--model_type', type=str, default='baseline_argument_model', help='')
    parser.add_argument('--encoder_model', type=str, default='bert-base-uncased', help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--max_epochs', type=int, default=1, help='')
    parser.add_argument('--monitor', type=str, default='val_f1', help='a metric determined to monitor the best model')
    parser.add_argument('--gpus', type=int, default=-1, help='')
    parser.add_argument('--cross_validation', type=int, default=1, help='make k-fold cross validation')
    
    args = parser.parse_args()

    return args 

    
def k_fold(k: int=10):
    if k == 0:
        raise ValueError('k should be larger than 0')
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    if not config.kfold_data_path.exists():
        config.kfold_data_path.mkdir()
    
    data1 = pd.read_csv(Path(config.train_file['arguments']).as_posix(), delimiter='\t')
    data2 = pd.read_csv(Path(config.validate_file['arguments']).as_posix(), delimiter='\t')
    data = pd.concat([data1, data2])
    l1 = pd.read_csv(Path(config.train_file['labels']).as_posix(), delimiter='\t')
    l2 = pd.read_csv(Path(config.validate_file['labels']).as_posix(), delimiter='\t')
    l = pd.concat([l1, l2])
    level1_1 = pd.read_csv(Path(config.train_file['level1-labels']).as_posix(), delimiter='\t')
    level1_2 = pd.read_csv(Path(config.validate_file['level1-labels']).as_posix(), delimiter='\t')
    level1 = pd.concat([level1_1, level1_2])
    index = 0
    for train, val in kf.split(data):
        train_argument_file = config.kfold_data_path/'arguments-training_{}.tsv'.format(index)
        validation_argument_file = config.kfold_data_path/'arguments-validation_{}.tsv'.format(index)
        train_label_file = config.kfold_data_path/'labels-training_{}.tsv'.format(index)
        train_level1_label_file = config.kfold_data_path/'level1-labels-training_{}.tsv'.format(index)
        validation_label_file = config.kfold_data_path/'labels-validation_{}.tsv'.format(index)
        validation_level1_label_file = config.kfold_data_path/'level1-labels-validation_{}.tsv'.format(index)
        data.iloc[train].to_csv(train_argument_file, sep='\t', index=False)
        l.iloc[train].to_csv(train_label_file, sep='\t', index=False)
        level1.iloc[train].to_csv(train_level1_label_file, sep='\t', index=False)
        data.iloc[val].to_csv(validation_argument_file, sep='\t', index=False)
        l.iloc[val].to_csv(validation_label_file, sep='\t', index=False)
        level1.iloc[val].to_csv(validation_level1_label_file, sep='\t', index=False)
        index += 1
        yield train_argument_file, train_label_file, train_level1_label_file, validation_argument_file, validation_label_file, validation_level1_label_file
        

def get_model_earlystopping_callback(monitor='val_f1', mode:Union['max', 'min']='max', min_delta=0.001):
    es_clb = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=3,
        verbose=True,
        mode=mode
        )
    return es_clb


def get_model_best_checkpoint_callback(dirpath='checkpoints', monitors='val_f1', mode:Union['max', 'min']='max'):
    s = '{' + '{}:.3f'.format(monitors)+ '}'
    bc_clb = ModelCheckpoint(
        filename='{epoch}-'+ s + '-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor=monitors,
        mode=mode
        )
    return  bc_clb


def save_model(trainer: pl.Trainer, default_root_dir=".", model_name='', timestamp=None):
    out_dir = default_root_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = os.path.join(out_dir, model_name + '_timestamp_' + str(timestamp) + '_final.ckpt')
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

def write_eval_performance(args: argparse.Namespace, eval_performance: Dict, out_file: Union[AnyStr, bytes, os.PathLike]):
    out_file = Path(out_file)
    if not out_file.parent.exists():
        out_file.parent.mkdir()
    json_data = dict()
    for key, value in args._get_kwargs():
        json_data[key] = [value]
    for key in eval_performance:
        json_data[key] = [eval_performance[key]]
    json_data = pd.DataFrame(json_data)
    if out_file.exists():
        data = pd.read_csv(out_file)
        json_data = pd.concat([data, json_data])
    json_data.to_csv(out_file, index=False)
    logging.info('Finished writing evaluation performance for {}'.format(out_file.as_posix()))

def write_test_results(test_results: List, out_file: Union[AnyStr, bytes, os.PathLike]):
    out_file = Path(out_file)
    if not out_file.parent.exists():
        out_file.parent.mkdir(parents=True)
    headers = get_header_from_label_file(config.validate_file['labels'])
    with open(str(out_file), 'w') as f:
        f.write("\t".join(headers))
        f.write("\n")
        for id, item in test_results:
            f.write(id+"\t")
            f.write("\t".join([str(int(field)) for field in item]))
            f.write("\n")

def validate_model(trainer: pl.Trainer, model: ArgumentModel, data_module: pl.LightningDataModule): 
    val_results = []
    preds = trainer.predict(model=model, dataloaders=data_module.val_dataloader())
    for argument_id, batch_result in preds:
        val_results.extend(list(zip(*(argument_id, batch_result))))
    return val_results
        
def test_model(trainer:pl.Trainer, model: ArgumentModel, data_module: pl.LightningDataModule):
    test_results = []
    preds = trainer.predict(model=model, dataloaders=data_module.test_dataloader())
    for argument_id, batch_result in preds:
        test_results.extend(list(zip(*(argument_id, batch_result))))
    return test_results

def generate_result_file_parent(trainer: pl.Trainer, args: argparse.Namespace, value_by_monitor: Dict):
    parent_name = Path("_".join(["{}={}".format(k, v) for k, v in args._get_kwargs()]))/('version_'+str(trainer.logger.version))
    name = "{}={}".format(args.monitor, str(value_by_monitor[args.monitor])) + ".tsv"
    return parent_name, name
    
def get_best_value(checkpoint_file: AnyStr, monitor: AnyStr='val_f1'):
    pattern = r'{}=(.*?)-'.format(monitor)
    val = re.findall(pattern, checkpoint_file)[0]
    return float(val)

def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return lr_monitor

def get_trainer(args):
    pl.seed_everything(42)
    callbacks = [get_model_earlystopping_callback(), get_model_best_checkpoint_callback(monitors=args.monitor)]


    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu', devices=args.gpus, max_epochs=args.max_epochs, callbacks=callbacks)
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=callbacks)

    logging.info('Finished create a trainer.')
    return trainer


def show_args(args):
    logging.info('run with these args:')
    log_info = "\n" + "\n".join(['{}: {}'.format(k, v) for k, v in args._get_kwargs()])
    logging.info(log_info)


def main(args: argparse.Namespace, train_arguments_file, train_label_file, train_level1_label_file, val_arguments_file, val_label_file, val_level1_label_file):
    trainer = get_trainer(args)
    adm = ArgumentDataModule.from_params(Params({
        'type': args.data_module_type,
        'reader': Params({
            'type': args.dataset_type,
            'encoder_model': args.encoder_model
        }),
        'batch_size': args.batch_size,
        'train_arguments_file': train_arguments_file,
        'train_label_file': train_label_file,
        'train_level1_label_file': train_level1_label_file,
        'val_arguments_file': val_arguments_file,
        'val_label_file': val_label_file,
        'val_level1_label_file': val_level1_label_file
    }))

    params = Params({
        'type': args.model_type,
        'encoder_model': args.encoder_model,
        'value_types': len(config.LABEL_NAME)
    })
    argument_model = ArgumentModel.from_params(params=params)
    trainer.fit(model=argument_model, datamodule=adm)
    _, best_checkpoint = save_model(trainer, model_name=args.model_type)
    logging.info('get best_checkpoint file: {}'.format(best_checkpoint))
    argument_model = load_model(ArgumentModel.by_name(args.model_type), model_file=best_checkpoint)
    logging.info('recording performance log....')
    trainer.validate(model=argument_model, datamodule=adm)
    value_by_monitor = argument_model.get_metric()
    write_eval_performance(args, value_by_monitor, config.performance_log)
    parent, file = generate_result_file_parent(trainer, args, value_by_monitor)
    val_out_file = config.output_path/parent/'val/'/file
    val_results = validate_model(trainer, argument_model, adm)
    logging.info('recording predictions of validation file....')
    write_test_results(test_results=val_results, out_file=val_out_file)
    #write performance metrics for future reference
    metric_file = config.output_path/parent/'metrics.tsv'
    write_eval_performance(args, value_by_monitor, metric_file)
    logging.info('recording predictions of test file....')
    test_results = test_model(trainer, argument_model, adm)
    parent, file = generate_result_file_parent(trainer, args, value_by_monitor)
    out_file = config.output_path/parent/file
    write_test_results(test_results=test_results, out_file=out_file)
    return metric_file, val_out_file, out_file

if __name__ == '__main__':
    args = parse_arguments()
    show_args(args)

    metric_files, test_preds_files = [], []
    if args.cross_validation == 1:
        can_vote_file = config.test_data_path/'non_cross_validation_can_vote_file.csv'
        metric_file, val_preds_file, _ = main(args, config.train_file['arguments'], config.train_file['labels'], config.train_file['level1-labels'], config.validate_file['arguments'], config.validate_file['labels'], config.validate_file['level1-labels'])
        metric_files.append(metric_file)
        test_preds_files.append(val_preds_file)
        voted_files = {
            'metric_files': metric_files,
            'test_preds_files': test_preds_files
        }
        if not Path(can_vote_file).exists():
            data = None
        else:
            data = pd.read_csv(can_vote_file)
        data = pd.concat([pd.DataFrame.from_dict(voted_files), data])
        data.to_csv(can_vote_file, index=False)
        all_voted_files = data
        out_file = config.test_data_path/'all_voted_labels.tsv'
        vote(all_voted_files['test_preds_files'].to_list(), all_voted_files['metric_files'].to_list(), out_file)
        ensemble_metric = calc_metric_by_file(label_file=config.validate_file['labels'], pred_file=out_file)
        logging.info('ensemble_metric is below: ')
        logging.info(ensemble_metric)
        
    else:
        out_file = config.test_data_path/'labels.tsv'
        for train_arguments_file, train_label_file, train_level1_label_file, val_arguments_file, val_label_file, val_level1_label_file in k_fold(args.cross_validation):
            metric_file, _, test_preds_file = main(args, train_arguments_file, train_label_file, train_level1_label_file, val_arguments_file, val_label_file, val_level1_label_file)
            metric_files.append(metric_file)
            test_preds_files.append(test_preds_file)
        vote(test_preds_files, metric_files, out_file)
        voted_files = {
            'metric_files': metric_files,
            'test_preds_files': test_preds_files
        }
        can_vote_file = config.test_data_path/'can_vote_file.csv'
        if not Path(can_vote_file).exists():
            data = None
        else:
            data = pd.read_csv(can_vote_file)
        data = pd.concat([pd.DataFrame.from_dict(voted_files), data])
        data.to_csv(can_vote_file, index=False)
        all_voted_files = data
        out_file = config.test_data_path/'all_voted_labels.tsv'
        vote(all_voted_files['test_preds_files'].to_list(), all_voted_files['metric_files'].to_list(), out_file)


    sys.exit(0)