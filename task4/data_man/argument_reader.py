# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


import pytorch_lightning as pl
from typing import AnyStr, Optional, Any, Union
import os
from task4.data_man.meta_data import read_arguments_from_file, read_labels_from_file
from torch.utils.data import Dataset
from task4.configuration.config import logging
from task4.configuration import config


class ArgumentsDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        self.instances = []
        self.labels = []

    def __getitem__(self, index: Any) -> Any:
        if index >= self.__len__():
            raise IndexError('index value must be not more than the maximum length.')
        if self.labels:
            return self.instances[index], self.labels[index]
        return self.instances[index], None

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


class ArgumentsDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.reader = ArgumentsDataset()
        
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
        return super().setup(stage)

    def collate_batch(self, batch):
        batch_size = len(batch)
        batch_ = list(zip(*batch))

        for argument_item, label_item in batch: 
            pass
    
if __name__ == "__main__":
    arg_dataset = ArgumentsDataset()
    arg_dataset.read_data(config.train_file['arguments'])
    logging.info(arg_dataset[0])
    arg_dataset.read_data(config.train_file['arguments'], config.train_file['labels'])
    logging.info(arg_dataset[0])
