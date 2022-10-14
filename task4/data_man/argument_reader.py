# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


import pytorch_lightning as pl
from typing import AnyStr, List, Optional, Any, Tuple, Union
import torch
import os
import torch.utils.data
from task4.data_man.meta_data import read_arguments_from_file, read_labels_from_file
from torch.utils.data import Dataset
from task4.configuration.config import logging
from task4.configuration import config
from task4.data_man.meta_data import ArgumentItem, LabelItem
from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from transformers import AutoTokenizer


class ArgumentsDataset(Dataset, Registrable):
    
    def __init__(
        self,
        encoder_model='bert-base-uncased'
        ) -> None:
        super().__init__()
        self.instances = []
        self.labels = []
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)

    def __getitem__(self, index: Any) -> Any:
        raise NotImplemented('')

    def encode_input(self, item):
        raise NotImplemented('')

    def __len__(self) -> int:
        return len(self.instances)

    def read_data(self, argument_file: Union[AnyStr, os.PathLike], label_file: Optional[Union[AnyStr, os.PathLike]]=None):
        self.instances = read_arguments_from_file(argument_file)
        if not label_file:
            self.labels = None
        else:
            self.labels = read_labels_from_file(label_file)
            if len(self.instances) != len(self.labels):
                raise ValueError('arguments lenght is not equal to label length.')


@ArgumentsDataset.register('baseline_argument_dataset')
class BaselineArgumentDataset(ArgumentsDataset):

    def encode_input(self, argument_item: ArgumentItem, label_item: LabelItem) -> Tuple[str, List[int], List[int], List[int], List[int]]:
        argument_id = argument_item.argument_id
        text = argument_item.conclusion + '[SEP]' + argument_item.stance + '[SEP]' + argument_item.premise
        outputs = self.tokenizer(text)
        input_ids, token_type_ids, attention_mask = outputs['input_ids'], outputs['token_type_ids'], outputs['attention_mask']
        if label_item:
            label_ids = label_item.label
        else:
            label_ids = None
        return argument_id, input_ids, token_type_ids, attention_mask, label_ids
        
    def __getitem__(self, index: Any):
        if index >= self.__len__():
            raise IndexError('index value must be not more than the maximum length.')
        if self.labels:
            return self.instances[index], self.labels[index]
        return self.instances[index], None


class ArgumentsDataModule(pl.LightningDataModule):

    def __init__(
        self, 
        reader: ArgumentsDataset,
        batch_size=16
        ):
        super().__init__()
        self.reader = reader
        self.batch_size = batch_size
        
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.reader.read_data(config.train_file['arguments'], config.train_file['labels'])
        if stage == 'validate':
            pass

        if stage == 'test':
            pass
            
        if stage == 'predict':
            pass

        self.stage = stage

    def collate_batch(self, batch):
        print(batch)
        batch_ = list(zip(*batch))
        print(batch_)

        #for argument_item, label_item in batch_: 
        #    print(argument_item, label_item)

        return batch_

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch)


if __name__ == "__main__":
    arg_dataset = ArgumentsDataset.from_params(
        Params({
            'type': 'baseline_argument_dataset',
            })
        )
    arg_dataset.read_data(config.train_file['arguments'])
    logging.info(arg_dataset[0])
    arg_dataset.read_data(config.train_file['arguments'], config.train_file['labels'])
    logging.info(arg_dataset[0])

    adm = ArgumentsDataModule(reader=arg_dataset, batch_size=2)
    adm.setup(stage='fit')
    for batch in adm.train_dataloader():
        print(batch)
        break