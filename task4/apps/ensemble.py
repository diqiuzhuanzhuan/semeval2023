# -*- coding: utf-8 -*-
# author: feynman
# email: diqiuzhuanzhuan@gmail.com

from typing import List, AnyStr, Dict, Union
import pandas as pd
import collections
import os
import numpy as np
from pathlib import Path


def vote_by_f1(predict_files: List[AnyStr], f1_files: List[AnyStr]) -> Dict[AnyStr, Dict[AnyStr, List]] :
    if len(predict_files) != len(f1_files):
        raise ValueError('{} should have the same length of {}'.format(predict_files, f1_files))
    poll_map = collections.defaultdict(collections.defaultdict)
    for pred_file, f1_file in zip(predict_files, f1_files):
        preds = pd.read_csv(pred_file, delimiter='\t', header=0)
        f1 = pd.read_csv(f1_file, delimiter=',', header=0).to_dict()
        for index, row in preds.iterrows():
            id = row['Argument ID']
            if not id in poll_map:
                poll_map[id] = collections.defaultdict(list)
            for k in row.keys():
                if k == 'Argument ID':
                    continue
                value = f1['val_' + k +'@f1'][0]
                if k not in poll_map[id]:
                    poll_map[id][k] = [0, 0]
                poll_map[id][k][row[k]] += value
    return poll_map

def convert_poll_map_to_result(poll_map: Dict[AnyStr, Dict[AnyStr, List]], out_file: Union[AnyStr, os.PathLike]):
    out_file = Path(out_file)
    out_dict = collections.defaultdict(list)
    for argument_id in poll_map:
        out_dict['Argument ID'].append(argument_id)
        for col_name in poll_map[argument_id]:
            v = np.argmax(poll_map[argument_id][col_name])
            out_dict[col_name].append(v)

    data = pd.DataFrame.from_dict(out_dict)
    if not out_file.parent.exists():
        out_file.parent.mkdir()
    data.to_csv(out_file, sep='\t', index=False)


if __name__ == "__main__":
    preds_files = ['val_f1=0.47.tsv']
    f1_files = ['metrics.tsv']
    poll_map = vote_by_f1(preds_files, f1_files)
    convert_poll_map_to_result(poll_map=poll_map, out_file='my.tsv')