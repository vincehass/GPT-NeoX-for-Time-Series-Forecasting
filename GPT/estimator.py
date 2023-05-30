"""
Copyright 2022 ServiceNow
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

>> The estimator object for TACTiS, which is used with GluonTS and PyTorchTS.
"""

from typing import Any, Dict, Optional, Iterable
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import as_stacked_batches
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import Cyclic
from gluonts.time_feature import time_features_from_frequency_str
import numpy as np
import torch
import torch.nn as nn
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.transform import (
    AddObservedValuesIndicator,
    CDFtoGaussianTransform,
    Chain,
    InstanceSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    RenameFields,
    TestSplitSampler,
    AddTimeFeatures,
    AsNumpyArray,
    DummyValueImputation,
    Transformation,
    cdf_to_gaussian_forward_transform,
)
# from pts import Trainer
# from pts.model import PyTorchEstimator
# from pts.model.utils import get_module_forward_input_names
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from .network import TACTiSPredictionNetwork, TACTiSTrainingNetwork
from GPT.lightning_module import TACTiSLightning
import pytorch_lightning as pl


PREDICTION_INPUT_NAMES = [
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]




class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1

        if window_size <= 0:
            return np.array([], dtype=int)

        indices = np.random.randint(window_size, size=1)
        return indices + a


class TACTiSlightEstimator(PyTorchLightningEstimator):
    """
    The compatibility layer between TACTiS and GluonTS / PyTorchTS.
    """

    def __init__(
        self,
        num_samples: int,
        model_parameters: Dict[str, Any],
        num_series: int,
        history_length: int,
        prediction_length: int,
        learning_rate:int,
        freq: str,
        cdf_normalization: bool = False,
        num_parallel_samples: int = 1,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ):
        """
        A PytorchTS wrapper for TACTiS

        Parameters:
        -----------
        model_parameters: Dict[str, Any]
            The parameters that will be sent to the TACTiS model.
        num_series: int
            The number of series in the multivariate data.
        history_length: int
            How many time steps will be sent to the model as observed.
        prediction_length: int
            How many time steps will be sent to the model as unobserved, to be predicted.
        freq: str
            The frequency of the series to be forecasted.
        trainer: Trainer
            A Pytorch-TS trainer object
        cdf_normalization: bool, default to False
            If set to True, then the data will be transformed using an estimated CDF from the
            historical data points, followed by the inverse CDF of a Normal(0, 1) distribution.
            Should not be used concurrently with the standardization normalization option in TACTiS.
        num_parallel_samples: int, default to 1
            How many samples to draw at the same time during forecast.
        """
        default_trainer_kwargs = {"max_epochs": 100}
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.model_parameters = model_parameters

        self.num_series = num_series
        self.num_samples = num_samples
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.freq = freq
        self.learning_rate = learning_rate
        self.cdf_normalization = cdf_normalization
        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.time_features = time_features_from_frequency_str("S")

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    # @classmethod
    # def derive_auto_fields(cls, train_iter):
    #     stats = calculate_dataset_statistics(train_iter)

    #     return {
    #         "num_feat_dynamic_real": stats.num_feat_dynamic_real,
    #         "num_feat_static_cat": len(stats.feat_static_cat),
    #         "cardinality": [len(cats) for cats in stats.feat_static_cat],
    #     }

    def create_training_network(self, device: torch.device) -> nn.Module:
        """
        Create the encapsulated TACTiS model which can be used for training.

        Parameters:
        -----------
        device: torch.device
            The device where the model parameters should be placed.

        Returns:
        --------
        model: nn.Module
            An instance of TACTiSTrainingNetwork.
        """
        return TACTiSTrainingNetwork(
            num_series=self.num_series,
            model_parameters=self.model_parameters,
        ).to(device=device)

    
    def create_instance_splitter(self, mode: str) -> Transformation:
        """
        Create and return the instance splitter needed for training, validation or testing.

        Parameters:
        -----------
        mode: str, "training", "validation", or "test"
            Whether to split the data for training, validation, or test (forecast)

        Returns
        -------
        Transformation
            The InstanceSplitter that will be applied entry-wise to datasets,
            at training, validation and inference time based on mode.
        """
        assert mode in ["training", "validation", "test"]

        if mode == "training":
            instance_sampler = SingleInstanceSampler(
                min_past=self.history_length,  # Will not pick incomplete sequences
                min_future=self.prediction_length,
            )
        elif mode == "validation":
            instance_sampler = SingleInstanceSampler(
                min_past=self.history_length,  # Will not pick incomplete sequences
                min_future=self.prediction_length,
            )
        elif mode == "test":
            # This splitter takes the last valid window from each multivariate series,
            # so any multi-window split must be done in the data definition.
            instance_sampler = TestSplitSampler()

        if self.cdf_normalization:
            normalize_transform = CDFtoGaussianTransform(
                cdf_suffix="_norm",
                target_field=FieldName.TARGET,
                target_dim=self.num_series,
                max_context_length=self.history_length,
                observed_values_field=FieldName.OBSERVED_VALUES,
            )
        else:
            normalize_transform = RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_norm",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_norm",
                }
            )

        return (
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=instance_sampler,
                past_length=self.history_length,
                future_length=self.prediction_length,
                time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
            )
            + normalize_transform
        )

        
    # def _create_instance_splitter(self, module: TACTiSLightning, mode: str):
    #     assert mode in ["training", "validation", "test"]

    #     instance_sampler = {
    #         "training": self.train_sampler,
    #         "validation": self.validation_sampler,
    #         "test": TestSplitSampler(),
    #     }[mode]

    #     return InstanceSplitter(
    #         target_field=FieldName.TARGET,
    #         is_pad_field=FieldName.IS_PAD,
    #         start_field=FieldName.START,
    #         forecast_start_field=FieldName.FORECAST_START,
    #         instance_sampler=instance_sampler,
    #         past_length=self.context_length,
    #         future_length=self.prediction_length,
    #         time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
    #         dummy_value=self.distr_output.value_in_support,
    #     )

    
    def create_transformation(self) -> Transformation:
        return Chain(
            [
                # FilterTransformation(lambda x: sum(abs(x[FieldName.TARGET])) > 0),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    imputation_method=DummyValueImputation(0.0),
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                AsNumpyArray(FieldName.FEAT_TIME, expected_ndim=2),
            ]
        )


    def create_predictor(
        self, transformation: Transformation, trained_network: nn.Module, device: torch.device
    ) -> PyTorchPredictor:
        """
        Create the predictor which can be used by GluonTS to do inference.

        Parameters:
        -----------
        transformation: Transformation
            The transformation to apply to the data prior to being sent to the model.
        trained_network: nn.Module
            An instance of TACTiSTrainingNetwork with trained parameters.
        device: torch.device
            The device where the model parameters should be placed.

        Returns:
        --------
        predictor: PyTorchPredictor
            The PyTorchTS predictor object.
        """
        prediction_network = TACTiSPredictionNetwork(
            num_series=self.num_series,
            model_parameters=self.model_parameters,
            prediction_length=self.prediction_length,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device=device)
        copy_parameters(trained_network, prediction_network)

        output_transform = cdf_to_gaussian_forward_transform if self.cdf_normalization else None
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            output_transform=output_transform,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=prediction_network,
            batch_size=self.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )



    def create_lightning_module(self) -> pl.LightningModule:
        return TACTiSLightning(
             num_series=self.num_samples,
             num_samples=self.num_series,
             model_parameters= self.model_parameters,
             learning_rate=self.learning_rate
        )
    


    def create_validation_data_loader(
        self,
        data: Dataset,
        module: TACTiSLightning,
        **kwargs,
    ) -> Iterable:
        instances = self.create_instance_splitter( "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )
    

    def create_training_data_loader(
        self,
        data: Dataset,
        module: TACTiSLightning,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self.create_instance_splitter( "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )
    



