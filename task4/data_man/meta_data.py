# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


from dataclasses import dataclass
from typing import Dict, List, Union
import os
import pandas as pd
from pathlib import Path


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
            label=json_dict['label']
        )
        

@dataclass 
class LabelDataset:
    """_summary_
    """
    labels_name: List[str]
    labels: List[LabelItem]
    
def read_arguments_from_file(file: Union[bytes, str, os.PathLike]) -> List[ArgumentItem]:
    data = pd.read_csv(Path(file).as_posix(), delimiter='\t')
    print('read {} arguments from {}.'.format(len(data), str(file)))
    return [ArgumentItem.from_dict(data.iloc[i]) for i in range(len(data))]

if __name__ == "__main__":
    from task4.configuration import config
    arguments = read_arguments_from_file(config.train_file['arguments'])
    print(arguments[0])
    