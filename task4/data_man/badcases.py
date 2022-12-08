# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import os, collections
from typing import AnyStr, Union
from task4.data_man.meta_data import read_labels_from_file, LABEL_NAME

def analyze_badcase(pred_file: Union[AnyStr, os.PathLike],  label_file: Union[AnyStr, os.PathLike]):
    pred_file = str(pred_file)
    label_file = str(label_file)
    pred_items = read_labels_from_file(pred_file)
    label_items = read_labels_from_file(label_file)
    if len(pred_items) != len(label_items):
        raise ValueError("the length is not equal.")
    stat_map = collections.defaultdict(set)
    for label_item, pred_item in zip(label_items, pred_items):
        for i, value in enumerate(pred_item.label):
            if pred_item.label[i] == label_item.label[i]:
                continue
            stat_map[LABEL_NAME[i]].add(label_item.argument_id)
    return stat_map
                
            
if __name__ == "__main__": 
    from task4.configuration import config
    label_file = config.validate_file['labels']
    pred_file = "./task4/data_man/val_f1=0.541.tsv"
    stat_map = analyze_badcase(pred_file, label_file)
    print(stat_map)
    