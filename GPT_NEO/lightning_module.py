

import pytorch_lightning as pl
from typing import Dict, Any
import torch
from Transformers.Model import TradingBot


class TradeBotLightning(pl.LightningModule):
    """
    Encapsulate the TradeBot model inside a Lightning Module shell.
    """

    def __init__(self, num_series: int, num_samples: int, model_parameters: Dict[str, Any], learning_rate: float):
        """
        Parameters:
        -----------
        num_series: int
            The number of independent series in the dataset the model will learn from.
        num_samples: int
            When forecasting, how many independent samples to generate for each time point.
        model_parameters: Dict[str, Any]
            The parameters to be sent to the TradeBot model.
            These will be logged as being the hyperparameters in the log.
        learning_rate: float
            The learning rate for the Adam optimizer.
        """
        super().__init__()

        self.num_samples = num_samples
        self.num_series = num_series
        self.learning_rate = learning_rate
        self.net = TradingBot(num_series=self.num_series, **model_parameters)
        self.save_hyperparameters("num_samples")
        self.save_hyperparameters("num_series")
        self.save_hyperparameters("model_parameters")
        self.save_hyperparameters("learning_rate")

    #def forward(self, hist_time: torch.Tensor, hist_value: torch.Tensor, pred_time: torch.Tensor) -> torch.Tensor:
        """
        Forecast the possible values for the series, at the given prediction time points.

        Parameters:
        -----------
        hist_time: Tensor [batch, time steps]
            A tensor containing the time steps associated with the values of hist_value.
        hist_value: Tensor [batch, series, time steps]
            A tensor containing the values that will be available at inference time.
        pred_time: Tensor [batch, time steps]
            A tensor containing the time steps associated with the values of pred_value.

        Returns:
        --------
        samples: torch.Tensor [batch, series, time steps, samples]
            Samples from the forecasted distribution.
        """
    def forward(self, *args, **kwargs):
        
        
        past_time_feat = kwargs["past_time_feat"]
        past_target = kwargs["past_target"]
        past_observed_values = kwargs["past_observed_values"]
        future_time_feat = kwargs["future_time_feat"]

        hist_time = past_time_feat
        hist_value = past_target
        pred_value = past_observed_values
        pred_time = future_time_feat        
        print('past_time_feat.shape',past_time_feat.shape)
        print('past_target.shape',past_target.shape)
        print('past_observed_values.shape',past_observed_values.shape)
        print('future_time_feat.shape',future_time_feat.shape)
        return self.net.sample(self.num_samples, hist_time, hist_value, pred_time)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Compute the loss function on a training batch.
        """
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        
        
        hist_time = past_time_feat
        hist_value = past_target
        pred_value = future_target
        pred_time = future_time_feat

        hist_value = hist_value.permute(0,-1,-2)
        pred_value = pred_value.permute(0,-1,-2)
        
        
        #Sanity check for time features
        # print('hist_time.shape',hist_time.shape)
        # print('hist_value.shape',hist_value.shape)
        # print('pred_time.shape',pred_time.shape)
        # print('pred_value.shape',pred_value.shape)
        #hist_time, hist_value, pred_time, pred_value = batch
        loss = self.net.loss(hist_time, hist_value, pred_time, pred_value)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Compute the loss function on a validation batch.
        """
        hist_time, hist_value, pred_time, pred_value = batch
        loss = self.net.loss(hist_time, hist_value, pred_time, pred_value)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Compute metrics on a test batch.
        Currently the only metric is the loss function.
        """
        hist_time, hist_value, pred_time, pred_value = batch
        loss = self.net.loss(hist_time, hist_value, pred_time, pred_value)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Forecast possible values for the series, over a test batch.
        """
        hist_time, hist_value, pred_time, _ = batch
        return self.forward(hist_time, hist_value, pred_time)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Create the optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
