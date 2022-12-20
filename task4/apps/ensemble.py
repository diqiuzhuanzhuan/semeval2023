# -*- coding: utf-8 -*-
# author: feynman
# email: diqiuzhuanzhuan@gmail.com

from typing import List, AnyStr
import pandas as pd
import collections


def vote_by_f1(predict_files: List[AnyStr], f1_files: List[AnyStr]):
    if len(predict_files) != len(f1_files):
        raise ValueError('{} should have the same length of {}'.format(predict_files, f1_files))
    poll_map = collections.defaultdict(collections.defaultdict(list))
    for pred_file, f1_file in zip(predict_files, f1_files):
        preds = pd.read_csv(pred_file, delimiter='\t', header=0)
        f1 = pd.read_csv(f1_file, delimiter='\t', header=0).to_dict()
        for index, row in preds.iterrows():
            id = row['Arugment ID']
            for k in row:
                if k == 'Argument ID':
                    continue
                value = f1[k][0]
                if k not in poll_map[id]:
                    poll_map[id][k] = [0, 0]
                poll_map[id][k][row[k]] += value
    return poll_map

if __name__ == "__main__":
    pass
              
        
        
