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

    
LABEL_NAME = ['Self-direction: thought', 'Self-direction: action', 'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources', 'Face', 'Security: personal',
    'Security: societal', 'Tradition', 'Conformity: rules', 'Conformity: interpersonal', 'Humility', 'Benevolence: caring', 'Benevolence: dependability', 'Universalism: concern',
    'Universalism: nature', 'Universalism: tolerance', 'Universalism: objectivity']


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
            label=[json_dict[key] for key in LABEL_NAME]
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
    print('read {} arguments from {}.'.format(len(data), str(file)))
    return [ArgumentItem.from_dict(data.iloc[i]) for i in range(len(data))]


def read_labels_from_file(file: Union[bytes, str, os.PathLike]) -> List[LabelItem]:
    """_summary_

    Args:
        file (Union[bytes, str, os.PathLike]): _description_

    Returns:
        List[LabelItem]: _description_
    """
    data = pd.read_csv(Path(file).as_posix(), delimiter='\t')
    print('read {} labels from {}.'.format(len(data), str(file)))
    return [LabelItem.from_dict(data.iloc[i]) for i in range(len(data))]


if __name__ == "__main__":
    from task4.configuration import config
    arguments = read_arguments_from_file(config.train_file['arguments'])
    print(arguments[0])
    labels = read_labels_from_file(config.train_file['labels'])
    print(labels[0])