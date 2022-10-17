# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com


from allennlp.common.registrable import Registrable
from allennlp.common.params import Params
import pytorch_lightning as pl
from typing import AnyStr, Any, Union, Optional
from transformers import AutoModel
import torch
import functools


class ArgumentModel(Registrable):
    pass

@ArgumentModel.register('baseline_argument_model') 
class BaselineArgumentModel(pl.LightningModule, ArgumentModel):
    
    def __init__(
        self, 
        encoder_model: AnyStr='bert-base-uncased',
        lr: float=1e-5,
        value_types=20
        ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        
        self.value_types = value_types
        self.multi_label_weight = torch.nn.Linear(self.encoder.config.hidden_size, self.value_types)
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        warmup_steps = self.warmup_steps
        def fn(warmup_steps, step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return 1.0

        def linear_warmup_decay(warmup_steps):
            return functools.partial(fn, warmup_steps)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


    def compute_loss(self, logits, targets):
        loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
        return loss

    def forward_step(self, batch):
        argument_id, input_ids, token_type_ids, attention_mask, label_ids = batch
        outputs = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]  # get [CLS]
        logits = self.multi_label_weight(cls_hidden_state)
        predict = torch.nn.Sigmoid()(logits)
        return_dict = {
            'logits': logits, 
            'predict': predict
        }
        if label_ids:
            loss = self.compute_loss(logits, label_ids)
            return_dict['loss'] = loss

        return return_dict
        

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)


if __name__ == '__main__':
    params = Params({
        'type': 'baseline_argument_model',
        'encoder_model': 'bert-base-uncased'         
    })
    argument_model = ArgumentModel.from_params(params=params)