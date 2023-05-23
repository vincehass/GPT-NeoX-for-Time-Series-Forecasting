import sys
sys.path.append('/Users/nhassen/Documents/ProjectQuant/MyRepos/Alternative/GPT_nex')
from pts.trainer import Trainer



import torch
from gluon.estimator import TradeBotEstimator
from gluon.dataset import generate_backtesting_datasets
from gluon.metrics import compute_validation_metrics
from gluon.plots import plot_four_forecasts
from gluonts.evaluation.backtest import make_evaluation_predictions

import warnings
warnings.filterwarnings("ignore")


history_factor = 3
backtest_id = 2

metadata, train_data, test_data = generate_backtesting_datasets("electricity_hourly", backtest_id, history_factor, use_cached=False)


for entry in list(train_data):
    entry["target"] = entry["target"][:20, :]
for entry in list(test_data):
    entry["target"] = entry["target"][:20, :]


estimator = TradeBotEstimator(
    model_parameters= {
        "rnn_decoder":{
            "dim_hidden_features":20,
            "num_layers":2,#32,
            "dim_output":168, #predict one day ahead for the next week 24*7
        },
    },
    
    num_series = list(train_data)[0]["target"].shape[0],
    input_dim = 4,
    gamma = 0.8,
    l_norm = 2,
    history_length = history_factor * metadata.prediction_length,
    prediction_length = metadata.prediction_length,
    freq = metadata.freq,
    trainer = Trainer(
        # input_dim = 4,
        # gamma = 0.8,
        # l_norm = 2,
        epochs = 100,
        batch_size = 256,
        num_batches_per_epoch = 512,
        learning_rate = 1e-3,
        weight_decay = 1e-4,
        maximum_learning_rate = 1e-3,
        clip_gradient = 1e3,
        device = torch.device("cuda"),
    ),
    cdf_normalization = False,
    num_parallel_samples = 100,
)


#predictor = estimator.train(train_data, test_data)


