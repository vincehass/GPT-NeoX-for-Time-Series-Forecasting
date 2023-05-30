import sys
sys.path.append('/Users/nhassen/Documents/ProjectQuant/MyRepos/test/GPT-NeoX-for-Time-Series-Forecasting/')

# Configuration
# import os
# import sys
# REPO_NAME = "tactis"
# def get_repo_basepath():
#     cd = os.path.abspath(os.curdir)
#     return cd[:cd.index(REPO_NAME) + len(REPO_NAME)]
# REPO_BASE_PATH = get_repo_basepath()
# sys.path.append(REPO_BASE_PATH)



import torch
from GPT.estimator import TradeBotLightEstimator
from gluon.dataset import generate_backtesting_datasets
from gluon.metrics import compute_validation_metrics
from gluon.plots import plot_four_forecasts
# from gluonts.evaluation.backtest import make_evaluation_predictions
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")


history_factor = 3
backtest_id = 2

metadata, train_data, test_data = generate_backtesting_datasets("electricity_hourly", backtest_id, history_factor, use_cached=False)


for entry in list(train_data):
    entry["target"] = entry["target"][:20, :]
for entry in list(test_data):
    entry["target"] = entry["target"][:20, :]


estimator = TradeBotLightEstimator(
    model_parameters= {
        "gamma":0.8,
        "l_norm": 2,
        "data_normalization":"standardization",
        "loss_normalization":"series",
        "series_embedding_dim":10,
        "input_encoder_layers":3,
        "input_encoding_normalization":True,
        "encoder": {
            "attention_layers":3,
            "attention_heads": 3,
            "attention_dim": 4,
            "attention_feedforward_dim": 12,
        },
        "quantile_decoder":{
             "min_u": 0.01,
             "max_u": 0.99,
            "attentional_quantile": {
                "attention_heads": 3,
                "attention_layers": 3,
                "attention_dim": 12,
                "mlp_layers": 3,
                "mlp_dim": 16,
                "resolution": 50,
            },
        }
    },
    
    num_series = list(train_data)[0]["target"].shape[0],
    num_samples = 10,
    history_length = history_factor * metadata.prediction_length,
    prediction_length = metadata.prediction_length,
    freq = metadata.freq,
    trainer_kwargs = dict(max_epochs=30, accelerator="cpu", devices=1),
    learning_rate = 1e-3,
    cdf_normalization = False,
    num_parallel_samples = 100,
)


predictor = estimator.train(training_data=train_data)