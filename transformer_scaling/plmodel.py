# Modified from https://github.com/kashif/pytorch-transformer-ts/blob/main/transformer/lightning_module.py

import torch
import pytorch_lightning as pl
from model import TransformerModel
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)


class TransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: TransformerModel= None,
        hyperparameters = None,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__()


        if not model:
            model = TransformerModel(
                activation="gelu", cardinality=[1], context_length=240, dim_feedforward=32, 
                distr_output=StudentTOutput(), dropout=0.1, embedding_dimension=None, freq=None, 
                input_size=1, lags_seq=[1, 2, 3, 4, 5, 6, 7, 24, 30], nhead=2, num_decoder_layers=6, num_encoder_layers=2, 
                num_feat_dynamic_real=6, num_feat_static_cat=1, num_feat_static_real=1, num_parallel_samples=100, prediction_length=24, scaling=True)
            
        self.save_hyperparameters("hyperparameters")
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
     
    def freq_mask(self, x, y, rate=0.1, dim=1):
        x_len = x.shape[dim]
        y_len = y.shape[dim]
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)
        if self.device == "gpu":
            m = torch.cuda.FloatTensor(xy_f.shape).uniform_() < rate
        else:
            m = torch.zeros_like(xy_f, dtype=torch.float).uniform_() < rate

        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)
        
        if x_len + y_len != xy.shape[dim]:
            xy = torch.cat([x[:,0:1, ...],  xy], 1)

        return torch.split(xy, [x_len, y_len], dim=dim)


    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        
        #past_target, future_target = self.freq_mask(x=batch["past_target"], y=batch["future_target"])
        #batch["past_target"] = past_target
        #batch["future_target"] = future_target

        #print(self.device, " ", torch.cuda.memory_allocated(self.device))
        train_loss = self(batch)
        #print(self.device, " ", torch.cuda.memory_allocated(self.device))

        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        
        
        with torch.inference_mode():
            val_loss = self(batch)
        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=True, prog_bar=True
        )
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
    
    def forward(self, batch):
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]

        #print(self.device, " ", torch.cuda.memory_allocated(self.device))
        transformer_inputs, scale, _ = self.model.create_network_inputs(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )
        #print(self.device, " ", torch.cuda.memory_allocated(self.device))
        params = self.model.output_params(transformer_inputs)
        #print(self.device, " ", torch.cuda.memory_allocated(self.device))
        distr = self.model.output_distribution(params, scale)

        loss_values = self.loss(distr, future_target)
    
        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)
