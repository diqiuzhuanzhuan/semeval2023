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

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

@ArgumentModel.register('baseline_argument_model') 
class BaselineArgumentModel(pl.LightningModule, ArgumentModel):
    
    def __init__(
        self, 
        encoder_model: AnyStr='bert-base-uncased',
        lr: float=1e-5,
        value_types=20,
        warmup_steps: int=1000,
        ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.value_types = value_types
        self.multi_label_weight = torch.nn.Linear(self.encoder.config.hidden_size, self.value_types)
        self.lr = lr
        self.warmup_steps = warmup_steps

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
        if label_ids is not None:
            loss = self.compute_loss(logits, label_ids)
            return_dict['loss'] = loss

        return return_dict

    def training_step(self, batch, batch_idx):
        outputs = self.forward_step(batch=batch)
        self.log_metrics({}, outputs['loss'], suffix='train', on_step=True, on_epoch=True)
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        outputs = self.forward_step(batch=batch)
        self.log_metrics({}, outputs['loss'], suffix='val', on_step=True, on_epoch=True)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.forward_step(batch=batch)
        return outputs
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        outputs = self.forward_step(batch=batch)
        return outputs


if __name__ == '__main__':
    params = Params({
        'type': 'baseline_argument_model',
        'encoder_model': 'bert-base-uncased'         
    })
    argument_model = ArgumentModel.from_params(params=params)