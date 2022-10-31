# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from torchmetrics import Metric
from typing import Optional, List
import collections
import torch


class ValueMetric(Metric):
    full_state_update: bool = True

    def __init__(self, id_to_type, rare_type) -> None:
        super().__init__()
        self.id_to_type = id_to_type
        self.rare_type = set(rare_type)
        self.tp = collections.defaultdict(int)
        self.tn = collections.defaultdict(int)
        self.fp = collections.defaultdict(int)
        self.fn = collections.defaultdict(int)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: List[List], target: List[List]):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy().tolist()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy().tolist()
        for p, t in zip(preds, target):
            self._update(preds=p, target=t)
            self.total += 1
    
    def _update(self, preds: List, target: List):
        for i, _ in enumerate(zip(preds, target)):
            if preds[i] == target[i]:
                if target[i] == 1:
                    self.tp[self.id_to_type[i]] += 1
                else:
                    self.tn[self.id_to_type[i]] += 1
            else: 
                if target[i] == 1:
                    self.fn[self.id_to_type[i]] += 1
                else:
                    self.fp[self.id_to_type[i]] += 1

    def compute(self):
        total_f1, acc = 0.0, 0.0
        total_number, acc_number = 0, 0
        score = dict()
        for i in range(len(self.id_to_type)):
            type_string = self.id_to_type[i]
            p_support = self.tp[type_string] + self.fp[type_string]
            if p_support == 0:
                precision = 0.0
            else:
                precision = self.tp[type_string] / p_support
            r_support = self.tp[type_string] + self.fn[type_string]
            if r_support == 0:
                recall = 0.0
            else:
                recall = self.tp[type_string] / r_support
            if precision * recall == 0:
                f1 = precision * recall
            else:
                f1 = (2 * precision * recall)/(precision + recall)
            score[type_string+'@f1'] = f1
            total_f1 += f1 / len(self.id_to_type)
            total_number += (self.tp[type_string] + self.fp[type_string] + self.tn[type_string] + self.fn[type_string])
            acc_number += self.tp[type_string] + self.tn[type_string]
        if total_number > 0:
            acc = acc_number / total_number
        score['f1'] = total_f1
        score['acc'] = acc
        return score
        
    
    def reset(self) -> None:
        self.fp.clear()
        self.fn.clear()
        self.tp.clear()
        self.tn.clear()
        return super().reset()
