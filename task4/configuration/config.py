# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com
import os
import pathlib
from typing import NamedTuple

root_path = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = root_path/'data'
log_path = root_path/'logs'
output_path = root_path/'output'

train_data_path = data_path/'training_data'



train_file = {
    'arguments': train_data_path/'arguments-training.tsv',
    'labels': train_data_path/'labels-training.tsv',
    'level1-labels': train_data_path/'level1-labels-training.tsv'
}