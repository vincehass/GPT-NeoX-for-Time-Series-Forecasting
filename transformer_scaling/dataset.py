from dataclasses import dataclass

from functools import lru_cache
import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np


from gluonts.dataset.common import Dataset, Cached
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    Chain,
    AddTimeFeatures,
    AddAgeFeature,
    ValidationSplitSampler,
    TestSplitSampler,
)

from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    minute_of_hour,
    hour_of_day,
    day_of_week,
    day_of_month,
    day_of_year,
)


time_features=[
        minute_of_hour,
        hour_of_day,
        day_of_week,
        day_of_month,
        day_of_year,
    ]


@lru_cache(10_000)
def as_period(val, freq):
    return pd.Period(val, freq)


@dataclass
class GluontsDataset(Dataset):
    def __init__(self, dataset, freq, prediction_length=24) -> None:
        super().__init__()
        transform = Chain([
             AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features,
                    pred_length=prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=prediction_length,
                    log_scale=True,
                ),
        ])

        self.dataset = Cached(transform.apply(dataset))
        self.freq = to_offset(freq)
        self.prediction_length = prediction_length

    def __iter__(self):
        for data in self.dataset:
            if len(data[FieldName.TARGET]) > self.prediction_length:
                yield {
                    FieldName.START: as_period(data[FieldName.START], self.freq),
                    FieldName.TARGET: data[FieldName.TARGET],
                    FieldName.FEAT_TIME: np.stack(data[FieldName.FEAT_TIME], 0),
                    FieldName.FEAT_AGE: np.stack(data[FieldName.FEAT_AGE], 0),
                    FieldName.ITEM_ID: data[FieldName.ITEM_ID],
                }

    def __len__(self):
        return len(self.dataset)
