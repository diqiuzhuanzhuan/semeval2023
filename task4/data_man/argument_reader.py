# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


from copy import deepcopy
import pytorch_lightning as pl
from typing import AnyStr, List, Optional, Any, Tuple, Union
import torch
import os
import torch.utils.data
from task4.data_man.meta_data import read_arguments_from_file, read_labels_from_file, read_level1_labels_from_file
from torch.utils.data import Dataset
from task4.configuration.config import logging
from task4.configuration import config
from task4.data_man.meta_data import ArgumentItem, LabelItem, Level1LabelItem
from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
from transformers import AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
class ArgumentsDataset(Dataset, Registrable):
    
    def __init__(
        self,
        encoder_model='bert-base-uncased'
        ) -> None:
        super().__init__()
        self.instances = []
        self.labels = []
        self.level1_labels = []
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model, add_prefix_space=True)

    def __getitem__(self, index: Any) -> Any:
        raise NotImplemented('')

    def encode_input(self, item):
        raise NotImplemented('')

    def __len__(self) -> int:
        return len(self.instances)

    def read_data(self, 
                  argument_file: Union[AnyStr, os.PathLike], 
                  label_file: Optional[Union[AnyStr, os.PathLike]]=None, 
                  level1_label_file: Optional[Union[AnyStr, os.PathLike]]=None):
        self.instances = read_arguments_from_file(argument_file)
        if not label_file:
            self.labels = None
            self.level1_labels = None
        else:
            self.labels = read_labels_from_file(label_file)
            self.level1_labels = read_level1_labels_from_file(level1_label_file)
            if len(self.instances) != len(self.labels) != len(self.level1_labels):
                raise ValueError('arguments lenght is not equal to label length.')


@ArgumentsDataset.register('baseline_argument_dataset')
class BaselineArgumentDataset(ArgumentsDataset):

    def encode_input(self, argument_item: ArgumentItem, label_item: LabelItem, level1_label_item: Level1LabelItem) -> Tuple[str, List[int], List[int], List[int], List[int]]:
        argument_id = argument_item.argument_id
        text = argument_item.premise + self.tokenizer.sep_token + argument_item.stance + self.tokenizer.sep_token + argument_item.conclusion
        outputs = self.tokenizer(text)
        input_ids, token_type_ids, attention_mask = outputs['input_ids'], outputs.get('token_type_ids', None), outputs['attention_mask']
        if token_type_ids is None:
            token_type_ids = [0] * len(input_ids)
        if label_item:
            label_ids = label_item.label
            level1_label_ids = level1_label_item.level1_label
        else:
            label_ids = None
            level1_label_ids = None
        return argument_id, input_ids, token_type_ids, attention_mask, label_ids, level1_label_ids
        
    def __getitem__(self, index: Any):
        if index >= self.__len__():
            raise IndexError('index value must be not more than the maximum length.')
        if self.labels:
            return self.encode_input(self.instances[index], self.labels[index], self.level1_labels[index])
        return self.encode_input(self.instances[index], None, None)


@ArgumentsDataset.register('premise_argument_dataset')
class PremiseArgumentDataset(BaselineArgumentDataset):
    def encode_input(self, argument_item: ArgumentItem, label_item: LabelItem, level1_label_item: Level1LabelItem) -> Tuple[str, List[int], List[int], List[int], List[int]]:
        argument_id = argument_item.argument_id
        text = argument_item.premise + self.tokenizer.sep_token
        outputs = self.tokenizer(text)
        input_ids, token_type_ids, attention_mask = outputs['input_ids'], outputs.get('token_type_ids', None), outputs['attention_mask']
        if token_type_ids is None:
            token_type_ids = [0] * len(input_ids)
        if label_item:
            label_ids = label_item.label
            level1_label_ids = level1_label_item.level1_label
        else:
            label_ids = None
            level1_label_ids = None
        return argument_id, input_ids, token_type_ids, attention_mask, label_ids, level1_label_ids


@ArgumentsDataset.register('rewrite_argument_dataset')
class RewriteArgumentDataset(BaselineArgumentDataset):
    def encode_input(self, argument_item: ArgumentItem, label_item: LabelItem, level1_label_item: Level1LabelItem) -> Tuple[str, List[int], List[int], List[int], List[int]]:
        argument_id = argument_item.argument_id
        if 'should not' in argument_item.conclusion:
            if argument_item.stance == 'in favor of':
                pass
            else:
                argument_item.conclusion = argument_item.conclusion.replace('should not', 'should')
            text = argument_item.premise + self.tokenizer.sep_token + argument_item.conclusion
        elif 'should' in argument_item.conclusion:
            if argument_item.stance == 'in favor of':
                pass
            else:
                argument_item.conclusion = argument_item.conclusion.replace('should', 'should not')
            text = argument_item.premise + self.tokenizer.sep_token + argument_item.conclusion
        elif 'do not need' in argument_item.conclusion or "don't need" in argument_item.conclusion:
            if argument_item.stance == 'in favor of':
                pass
            else:
                argument_item.conclusion = argument_item.conclusion.replace('do not need', 'need')
                argument_item.conclusion = argument_item.conclusion.replace("don't need", 'need')
            text = argument_item.premise + self.tokenizer.sep_token + argument_item.conclusion
        elif 'need' in argument_item.conclusion or 'needs' in argument_item.conclusion:
            if argument_item.stance == 'in favor of':
                pass
            else:
                argument_item.conclusion = argument_item.conclusion.replace('need', 'do not need')
                argument_item.conclusion = argument_item.conclusion.replace('needs', 'do not need')
            text = argument_item.premise + self.tokenizer.sep_token + argument_item.conclusion
        else:
            text = argument_item.premise
        outputs = self.tokenizer(text)
        input_ids, token_type_ids, attention_mask = outputs['input_ids'], outputs.get('token_type_ids', None), outputs['attention_mask']
        if token_type_ids is None:
            token_type_ids = [0] * len(input_ids)
        if label_item:
            label_ids = label_item.label
            level1_label_ids = level1_label_item.level1_label
        else:
            label_ids = None
            level1_label_ids = None
        return argument_id, input_ids, token_type_ids, attention_mask, label_ids, level1_label_ids


@ArgumentsDataset.register('label_match_argument_dataset')
class LabelMatchArgumentDataset(BaselineArgumentDataset):
    def encode_input(self, argument_item: ArgumentItem, label_item: LabelItem) -> Tuple[str, List[int], List[int], List[int], List[int]]:
        argument_id = argument_item.argument_id
        text = argument_item.premise + self.tokenizer.sep_token
        outputs = self.tokenizer(text)
        input_ids, token_type_ids, attention_mask = outputs['input_ids'], outputs.get('token_type_ids', None), outputs['attention_mask']
        if token_type_ids is None:
            token_type_ids = [0] * len(input_ids)
        if label_item:
            label_ids = label_item.label
        else:
            label_ids = None
        return argument_id, input_ids, token_type_ids, attention_mask, label_ids
    
        
class ArgumentDataModule(pl.LightningDataModule, Registrable):
    pass


@ArgumentDataModule.register('baseline_argument_data_module')
class BaselineArgumentDataModule(ArgumentDataModule):

    def __init__(
        self, 
        reader: ArgumentsDataset,
        train_arguments_file: AnyStr,
        train_label_file: AnyStr,
        train_level1_label_file: AnyStr,
        val_arguments_file: AnyStr,
        val_label_file: AnyStr,
        val_level1_label_file: AnyStr,
        batch_size=16
        ):
        super().__init__()
        self.reader = reader
        self.batch_size = batch_size
        self.train_arguments_file = train_arguments_file
        self.train_label_file = train_label_file
        self.train_level1_label_file = train_level1_label_file
        self.val_arguments_file = val_arguments_file
        self.val_label_file = val_label_file
        self.val_level1_label_file = val_level1_label_file
        
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            pass
        if stage == 'validate':
            pass

        if stage == 'test':
            pass 
        if stage == 'predict':
            pass

        self.stage = stage


    def collate_batch(self, batch):
        batch_size = len(batch)
        batch_ = list(zip(*batch))
        argument_id, input_ids, token_type_ids, attention_mask, label_ids, level1_label_ids = batch_
        max_len = max([len(_) for _ in input_ids])
        input_ids_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        token_type_ids_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        attention_mask_tensor = torch.empty(size=[batch_size, max_len], dtype=torch.long).fill_(0)
        if all(label_ids):
            max_label_len = max([len(_) for _ in label_ids])
            label_ids_tensor = torch.empty(size=[batch_size, max_label_len], dtype=torch.float).fill_(0)
            max_level1_label_len = max([len(_) for _ in level1_label_ids])
            level1_label_ids_tensor = torch.empty(size=[batch_size, max_level1_label_len], dtype=torch.float).fill_(0)
        else:
            label_ids_tensor = None
            level1_label_ids_tensor = None
        for i in range(batch_size):
            available_length = len(input_ids[i])
            input_ids_tensor[i][0:available_length] = torch.tensor(input_ids[i], dtype=torch.long)
            token_type_ids_tensor[i][0:available_length] = torch.tensor(token_type_ids[i], dtype=torch.long)
            attention_mask_tensor[i][0:available_length] = torch.tensor(attention_mask[i], dtype=torch.long)
            if label_ids_tensor is not None:
                label_ids_tensor[i] = torch.tensor(label_ids[i], dtype=torch.float)
                level1_label_ids_tensor[i] = torch.tensor(level1_label_ids[i], dtype=torch.float)

        return argument_id, input_ids_tensor, token_type_ids_tensor, attention_mask_tensor, label_ids_tensor, level1_label_ids_tensor

    def train_dataloader(self):
        self.reader.read_data(self.train_arguments_file, self.train_label_file, self.train_level1_label_file)
        train_reader = deepcopy(self.reader)
        return torch.utils.data.DataLoader(train_reader, batch_size=self.batch_size, collate_fn=self.collate_batch, shuffle=True, num_workers=8)

    def val_dataloader(self):
        self.reader.read_data(self.val_arguments_file, self.val_label_file, self.val_level1_label_file)
        val_reader = deepcopy(self.reader)
        return torch.utils.data.DataLoader(val_reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)

    def test_dataloader(self):
        self.reader.read_data(config.test_file['arguments'])
        test_reader = deepcopy(self.reader)
        return torch.utils.data.DataLoader(test_reader, batch_size=self.batch_size, collate_fn=self.collate_batch, num_workers=8)
    
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
    arg_dataset.read_data(config.train_file['arguments'], config.train_file['labels'], config.train_file['level1-labels'])
    logging.info(arg_dataset[0])

    adm = ArgumentDataModule.from_params(Params({
        'type': 'baseline_argument_data_module',
        'reader': Params({
            'type': 'rewrite_argument_dataset'    
        }),
        'batch_size': 2,
        'train_arguments_file': config.train_file['arguments'],
        'train_label_file': config.train_file['labels'], 
        'train_level1_label_file': config.train_file['level1-labels'],
        'val_arguments_file': config.validate_file['arguments'],
        'val_label_file': config.validate_file['labels'],
        'val_level1_label_file': config.validate_file['level1-labels']
    }))
    adm.setup(stage='fit')
    t_train = adm.train_dataloader()
    for batch in t_train:
        pass
    t_val = adm.val_dataloader()
    for batch in t_val:
        print(batch)
        break
    for batch in t_train:
        print(batch)
        break