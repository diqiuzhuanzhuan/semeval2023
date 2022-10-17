# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
import pytorch_lightning as pl
from typing import AnyStr, Any, Union, Optional
from transformers import AutoModel


class ArgumentModel(Registrable):
    pass

@ArgumentModel.register('baseline_argument_model') 
class BaselineArgumentModel(pl.LightningModule, ArgumentModel):
    
    def __init__(
        self, 
        encoder_model: AnyStr='bert-base-uncased'
        ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)


if __name__ == '__main__':
    params = Params({
        'type': 'baseline_argument_model',
        'encoder_model': 'bert-base-uncased'         
    })
    argument_model = ArgumentModel.from_params(params=params)