# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from torchmetrics import Metric
from typing import Optional, List
import collections
import torch


class EventTypeMetric(Metric):

    def __init__(self, id_to_type, rare_type, compute_on_step: Optional[bool] = None) -> None:
        super().__init__(compute_on_step)
        self.event_ignore_index = 2
        self.id_to_type = id_to_type
        self.rare_type = set(rare_type)
        self.tp = collections.defaultdict(int)
        self.tn = collections.defaultdict(int)
        self.fp = collections.defaultdict(int)
        self.fn = collections.defaultdict(int)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: List[List], target: List[List]):
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
                if preds[i] == 1:
                    self.fn[self.id_to_type[i]] += 1
                else:
                    self.fp[self.id_to_type[i]] += 1

    def compute(self):
        for i in range(len(self.id_to_type)):
            type_string = self.id_to_type[i]
            p_support = self.tp[type_string] + self.fp[type_string]
            if p_support == 0:
                precision = 0
            else:
                precision = self.tp[type_string] / p_support
            r_support = self.tp[type_string] + self.fn[type_string]
            if r_support == 0:
                recall = 0
            else:
                recall = self.tp[type_string] / r_support
            if precision * recall == 0:
                f1 = precision * recall
            else:
                f1 = (2 * precision * recall)/(precision + recall)
        score = {
            'f1': f1
        }
        return score
        
    
    def reset(self) -> None:
        self.fp.clear()
        self.fn.clear()
        self.tp.clear()
        self.tn.clear()
        return super().reset()