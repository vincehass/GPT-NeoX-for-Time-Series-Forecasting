#!/usr/bin/env python

import os, time, sys
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

os.environ["NODE_RANK"] = str(rank)
os.environ["LOCAL_RANK"] = str(rank % 6)


import lightning.pytorch as pl
sys.modules['pytorch_lightning'] = pl


from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from dataset import GluontsDataset, time_features
from estimator import TransformerEstimator


import config

dataset = get_dataset("electricity")
test = GluontsDataset(dataset.test, dataset.metadata.freq, 24)


_, estimator_opt, feature_opt, checkpoint_opt, logger_opt = config.read_config()
extra_params = {
    "freq": dataset.metadata.freq,
    "cardinality": [int(i.cardinality) for i in dataset.metadata.feat_static_cat],
    "embedding_dimension": [feature_opt["staticFeatureEmbDim"]] * len(dataset.metadata.feat_static_cat)
}


estimator = TransformerEstimator(
    **estimator_opt,
    **extra_params,
    time_features = time_features)


checkpoint_path = checkpoint_opt["inferenceCheckpoint"] if checkpoint_opt and "inferenceCheckpoint" in checkpoint_opt else None
predictor = estimator.get_predictor(checkpoint_path)



#Evaluation
if rank == 0:
    forecast_it, ts_it = make_evaluation_predictions(dataset=test, predictor=predictor)
    forecast, ts = list(forecast_it), list(ts_it)

    evaluator = Evaluator(num_workers=None)
    agg_metrics, _ = evaluator(iter(ts), iter(forecast))
    print(agg_metrics['mean_wQuantileLoss'])
