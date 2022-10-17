# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

import pytorch_lightning as pl
from task4.modeling.model import ArgumentModel
from allennlp.common.params import Params
from task4.data_man.argument_reader import ArgumentDataModule

if __name__ == "__main__":

    trainer = pl.Trainer(max_epochs=1)
    adm = ArgumentDataModule.from_params(Params({
        'type': 'baseline_argument_data_module',
        'reader': Params({
            'type': 'baseline_argument_dataset'    
        }),
        'batch_size': 2
    }))

    params = Params({
        'type': 'baseline_argument_model',
        'encoder_model': 'bert-base-uncased'         
    })
    argument_model = ArgumentModel.from_params(params=params)
    trainer.fit(model=argument_model, datamodule=adm)
