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
            return self.encode_input(self.instances[index], self.labels[index])
        return self.encode_input(self.instances[index], None)

        
class ArgumentDataModule(pl.LightningDataModule, Registrable):
    pass


@ArgumentDataModule.register('baseline_argument_data_module')
class BaselineArgumentDataModule(ArgumentDataModule):

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
            self.reader.read_data(config.validate_file['arguments'], config.validate_file['labels'])

        if stage == 'test':
            self.reader.read_data(config.test_file['arguments'], config.test_file['labels'])
            
        if stage == 'predict':
            pass

        self.stage = stage


    def collate_batch(self, batch):
        batch_size = len(batch)
        batch_ = list(zip(*batch))
        argument_id, input_ids, token_type_ids, attention_mask, label_ids = batch_
        max_len = max([len(_) for _ in input_ids])
        input_ids_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        token_type_ids_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        attention_mask_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        if label_ids:
            max_label_len = max([len(_) for _ in label_ids])
            label_ids_tensor = torch.empty(size=[batch_size, max_label_len], dtype=torch.float).fill_(0)
        else:
            label_ids_tensor = None
        for i in range(batch_size):
            available_length = len(input_ids[i])
            input_ids_tensor[i][0:available_length] = torch.tensor(input_ids[i], dtype=torch.long)
            token_type_ids_tensor[i][0:available_length] = torch.tensor(token_type_ids[i], dtype=torch.long)
            attention_mask_tensor[i][0:available_length] = torch.tensor(attention_mask[i], dtype=torch.long)
            if label_ids_tensor is not None:
                label_ids_tensor[i] = torch.tensor(label_ids[i], dtype=torch.float)

        return argument_id, input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, label_ids_tensor

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)


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

    adm = ArgumentDataModule.from_params(Params({
        'type': 'baseline_argument_data_module',
        'reader': Params({
            'type': 'baseline_argument_dataset'    
        }),
        'batch_size': 2
    }))
    adm.setup(stage='fit')
    for batch in adm.train_dataloader():
        print(batch)
        break