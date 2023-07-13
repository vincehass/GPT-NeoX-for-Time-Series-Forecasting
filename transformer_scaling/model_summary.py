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




from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
#from lightning.pytorch.plugins.environments import MPIEnvironment
from gluonts.dataset.repository.datasets import get_dataset

from dataset import GluontsDataset, time_features
from estimator import TransformerEstimator


import config






dataset = get_dataset("electricity")
train = GluontsDataset(dataset.test, dataset.metadata.freq, 24)

trainer_opt, estimator_opt, log_name = config.read_config()


seed = 1
logger = CSVLogger("logs_aug", name=log_name)

if "strategy" in trainer_opt and trainer_opt["strategy"] == "ddp":
    trainer_opt["strategy"] = DDPStrategy(find_unused_parameters=False)

    
trainer_kwargs=dict(**trainer_opt, logger=logger, gradient_clip_val=5) #enable_checkpointing=False


estimator = TransformerEstimator(
    **estimator_opt,
    time_features = time_features, 
    trainer_kwargs=trainer_kwargs)


model = estimator.create_lightning_module().model
d_model = model.input_size * len(model.lags_seq) + model._number_of_features
print(model.transformer)
print("Number of parameters: ", sum([param.nelement() for param in model.parameters()]))




