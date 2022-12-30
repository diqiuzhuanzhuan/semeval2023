# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


from dataclasses import dataclass
from typing import Dict, List, Union
import os
import pandas as pd
from pathlib import Path
from task4.configuration import config
from task4.configuration.config import logging


@dataclass
class ArgumentItem:
    """_summary_
    """
    argument_id: str
    conclusion: str
    stance: str
    premise: str

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return ArgumentItem(
            argument_id=json_dict['Argument ID'],
            conclusion=json_dict['Conclusion'],
            stance=json_dict['Stance'],
            premise=json_dict['Premise']
        )


@dataclass
class ArgumentDataset:
    """_summary_
    """
    dataset_path: str
    arguments: List[ArgumentItem]

    

def get_id_to_type():
    return_map = dict()
    for i, ele in enumerate(config.LABEL_NAME):
        return_map[i] = ele
    return return_map

@dataclass        
class LabelItem:
    """_summary_
    """
    argument_id: str
    label: List[int]

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return LabelItem(
            argument_id=json_dict['Argument ID'],
            label=[json_dict[key] for key in config.LABEL_NAME]
        )

@dataclass
class Level1LabelItem:
    """_summary_

    Returns:
        _type_: _description_
    """
    argument_id: str
    level1_label: List[int]
    
    @classmethod
    def from_dict(cls, json_dict: Dict):
        return Level1LabelItem(
            argument_id=json_dict['Argument ID'],
            level1_label=[json_dict[key] for key in config.LEVEL1_LABEL_NAME]
        )
        

@dataclass 
class LabelDataset:
    """_summary_
    """
    labels_name: List[str]
    labels: List[LabelItem]

    
def read_arguments_from_file(file: Union[bytes, str, os.PathLike]) -> List[ArgumentItem]:
    """_summary_

    Args:
        file (Union[bytes, str, os.PathLike]): _description_

    Returns:
        List[ArgumentItem]: _description_
    """
    data = pd.read_csv(Path(file).as_posix(), delimiter='\t')
    logging.info('read {} arguments from {}.'.format(len(data), str(file)))
    return [ArgumentItem.from_dict(data.iloc[i]) for i in range(len(data))]


def read_labels_from_file(file: Union[bytes, str, os.PathLike]) -> List[LabelItem]:
    """_summary_

    Args:
        file (Union[bytes, str, os.PathLike]): _description_

    Returns:
        List[LabelItem]: _description_
    """
    data = pd.read_csv(Path(file).as_posix(), delimiter='\t')
    logging.info('read {} labels from {}.'.format(len(data), str(file)))
    return [LabelItem.from_dict(data.iloc[i]) for i in range(len(data))]


def read_level1_labels_from_file(file: Union[bytes, str, os.PathLike]) -> List[Level1LabelItem]:
    """_summary_

    Args:
        file (Union[bytes, str, os.PathLike]): _description_

    Returns:
        List[Level1LabelItem]: _description_
    """
    data = pd.read_csv(Path(file).as_posix(), delimiter='\t')
    logging.info('read {} labels from {}.'.format(len(data), str(file)))
    return [Level1LabelItem.from_dict(data.iloc[i]) for i in range(len(data))]


def get_header_from_label_file(file: Union[bytes, str, os.PathLike]) -> List[str]:
    """_summary_

    Args:
        file (Union[bytes, str, os.PathLike]): _description_

    Returns:
        List[str]: _description_
    """
    data = pd.read_csv(Path(file).as_posix(), delimiter='\t')
    return list(data)


if __name__ == "__main__":
    from task4.configuration import config
    arguments = read_arguments_from_file(config.train_file['arguments'])
    print(arguments[0])
    labels = read_labels_from_file(config.train_file['labels'])
    print(labels[0])
    level1_labels = read_level1_labels_from_file(config.train_file['level1-labels'])
    print(level1_labels[0])