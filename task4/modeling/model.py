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
from task4.configuration import config
from task4.metric.value_metric import ValueMetric
from task4.data_man.meta_data import get_id_to_type
from task4.modeling.some_loss import ResampleLoss



def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def linear_warmup_decay(warmup_steps):
    return functools.partial(fn, warmup_steps)

class ArgumentModel(Registrable, pl.LightningModule):
    lr = 1e-5
    warmup_steps = 1000

    def log_metrics(self, pred_results, loss=0.0, suffix='', on_step=False, on_epoch=True):
        for key in pred_results:
            self.log(suffix + key, pred_results[key], on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

        self.log(suffix + 'loss', loss, on_step=on_step, on_epoch=on_epoch, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        warmup_steps = self.warmup_steps

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def forward_step(self, batch):
        raise NotImplementedError()


@ArgumentModel.register('baseline_argument_model') 
class BaselineArgumentModel(ArgumentModel):
    
    def __init__(
        self, 
        encoder_model: AnyStr='bert-base-uncased',
        lr: float=1e-5,
        value_types: int=20,
        warmup_steps: int=1000,
        ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model)
        self.value_types = value_types
        self.multi_label_weight = torch.nn.Linear(self.encoder.config.hidden_size, self.value_types)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.metric = ValueMetric(id_to_type=get_id_to_type(), rare_type=[])
        self.save_hyperparameters({
            'encoder_model': encoder_model, 
            'lr': lr, 
            'value_types': value_types, 
            'warmup_steps': warmup_steps
            })

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
        predict = (torch.nn.Sigmoid()(logits) > 0.5) * 1.0
        return_dict = {
            'logits': logits, 
            'predict': predict
        }
        if label_ids is not None:
            loss = self.compute_loss(logits, label_ids)
            return_dict['loss'] = loss
            self.metric.update(preds=predict, target=label_ids)
            return_dict['metric'] = self.metric.compute()

        return return_dict

    def training_step(self, batch, batch_idx):
        outputs = self.forward_step(batch=batch)
        self.log_metrics(outputs['metric'], outputs['loss'], suffix='train_', on_step=True, on_epoch=False)
        return {'loss': outputs['loss']}

    def on_train_epoch_start(self) -> None:
        self.metric.reset()
        return super().on_train_epoch_start()

    def training_epoch_end(self, outputs):
        average_loss = torch.mean(torch.tensor([item['loss'] for item in outputs], device=self.device))
        metric = self.metric.compute()
        self.log_metrics(metric, average_loss, suffix='train_', on_step=False, on_epoch=True)
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        outputs = self.forward_step(batch=batch)
        self.log_metrics(outputs['metric'], outputs['loss'], suffix='val_', on_step=True, on_epoch=False)
        return {'loss': outputs['loss']}

    def on_validation_epoch_start(self) -> None:
        self.metric.reset()
        return super().on_validation_epoch_start()
    
    def validation_epoch_end(self, outputs) -> None:
        average_loss = torch.mean(torch.tensor([item['loss'] for item in outputs], device=self.device))
        metric = self.metric.compute()
        self.log_metrics(metric, average_loss, suffix='val_', on_step=False, on_epoch=True)
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        outputs = self.forward_step(batch=batch)
        return outputs
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        outputs = self.forward_step(batch=batch)
        return outputs['predict']

    def predict_tags(self, batch: Any) -> Any:
        argument_id, input_ids, token_type_ids, attention_mask, label_ids = batch
        outputs = self.forward_step(batch=batch)
        return argument_id, outputs['predict'].numpy().tolist()


@ArgumentModel.register('focal_loss_argument_model') 
class FocalLossArgumentModel(BaselineArgumentModel):

    def __init__(
        self, 
        encoder_model: AnyStr = 'bert-base-uncased', 
        lr: float = 0.00001, 
        value_types: int = 20, 
        warmup_steps: int = 1000
        ) -> None:
        super().__init__(encoder_model, lr, value_types, warmup_steps)

    def compute_loss(self, logits, targets):
        class_freq = torch.tensor(config.label_freq, device=self.device)
        train_num = torch.tensor(config.train_num, device=self.device)
        loss_func = ResampleLoss(
            reweight_func=None, 
            loss_weight=1.0,
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(),
            class_freq=class_freq, 
            train_num=train_num
            )
        loss = loss_func(logits, targets)
        return loss

@ArgumentModel.register('class_balanced_loss_argument_model') 
class ClassBalancedLossArgumentModel(BaselineArgumentModel):

    def __init__(
        self, 
        encoder_model: AnyStr = 'bert-base-uncased', 
        lr: float = 0.00001, 
        value_types: int = 20, 
        warmup_steps: int = 1000
        ) -> None:
        super().__init__(encoder_model, lr, value_types, warmup_steps)

    def compute_loss(self, logits, targets):
        class_freq = torch.tensor(config.label_freq, device=self.device)
        train_num = torch.tensor(config.train_num, device=self.device)
        loss_func = ResampleLoss(
            reweight_func='CB', 
            loss_weight=10.0,
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(),
            CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
            class_freq=class_freq, 
            train_num=train_num) 
        loss = loss_func(logits, targets)
        return loss

@ArgumentModel.register('rbce_focal_loss_argument_model')        
class RbceFocalLossArgumentModel(BaselineArgumentModel):

    def __init__(
        self, 
        encoder_model: AnyStr = 'bert-base-uncased', 
        lr: float = 0.00001, 
        value_types: int = 20, 
        warmup_steps: int = 1000
        ) -> None:
        super().__init__(encoder_model, lr, value_types, warmup_steps)

    def compute_loss(self, logits, targets):
        class_freq = torch.tensor(config.label_freq, device=self.device)
        train_num = torch.tensor(config.train_num, device=self.device)
        loss_func = ResampleLoss(
            reweight_func='rebalance', 
            loss_weight=1.0, 
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
            class_freq=class_freq, 
            train_num=train_num
            )
        loss = loss_func(logits, targets)
        return loss


@ArgumentModel.register('ntr_focal_loss_argument_model') 
class NtrFocalLossArgumentModel(BaselineArgumentModel):

    def compute_loss(self, logits, targets):
        class_freq = torch.tensor(config.label_freq, device=self.device)
        train_num = torch.tensor(config.train_num, device=self.device)
        loss_func = ResampleLoss(
            reweight_func=None, 
            loss_weight=1.0,
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=2.0),
            class_freq=class_freq,
            train_num=train_num
            )
        loss = loss_func(logits, targets)
        return loss

@ArgumentModel.register('db_no_focal_loss_argument_model')
class DbNoFocalLossArgumentModel(BaselineArgumentModel):

    def compute_loss(self, logits, targets):
        class_freq = torch.tensor(config.label_freq, device=self.device)
        train_num = torch.tensor(config.train_num, device=self.device)
        loss_func = ResampleLoss(
            reweight_func='rebalance', 
            loss_weight=0.5,
            focal=dict(focal=False, alpha=0.5, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=2.0),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
            class_freq=class_freq,
            train_num=train_num
            )
        loss = loss_func(logits, targets)
        return loss

@ArgumentModel.register('class_balanced_ntr_loss_argument_model')
class ClassBalancedNtrLossArgumentModel(BaselineArgumentModel):

    def compute_loss(self, logits, targets):
        class_freq = torch.tensor(config.label_freq, device=self.device)
        train_num = torch.tensor(config.train_num, device=self.device)
        loss_func = ResampleLoss(
            reweight_func='CB', 
            loss_weight=10.0,
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=2.0),
            CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
            class_freq=class_freq,
            train_num=train_num
            )
        loss = loss_func(logits, targets)
        return loss

@ArgumentModel.register('distribution_balanced_loss_argument_model')
class DistributionBalancedLossArgumentModel(BaselineArgumentModel):

    def compute_loss(self, logits, targets):
        class_freq = torch.tensor(config.label_freq, device=self.device)
        train_num = torch.tensor(config.train_num, device=self.device)
        loss_func = ResampleLoss(
            reweight_func='rebalance', 
            loss_weight=1.0,
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=2.0),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
            class_freq=class_freq,
            train_num=train_num
            )
        loss = loss_func(logits, targets)
        return loss


if __name__ == '__main__':
    params = Params({
        'type': 'baseline_argument_model',
        'encoder_model': 'bert-base-uncased'         
    })
    argument_model = ArgumentModel.from_params(params=params)