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

from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from dataset import GluontsDataset, time_features
from estimator import TransformerEstimator


import config



dataset = get_dataset("electricity")
train = GluontsDataset(dataset.train, dataset.metadata.freq, 24)

trainer_opt, estimator_opt, feature_opt, checkpoint_opt, logger_opt = config.read_config()


seed = 1
logger = CSVLogger(**logger_opt)

class CustomDDPStrategy(DDPStrategy):
    def configure_ddp(self):
        super(CustomDDPStrategy, self).configure_ddp()
        self.model._set_static_graph()



        
if "strategy" in trainer_opt and trainer_opt["strategy"] == "ddp":
    trainer_opt["strategy"] = CustomDDPStrategy(find_unused_parameters=False) #static_graph = False for PT 2.0


trainer_kwargs=dict(**trainer_opt, logger = logger)
extra_params = {
    "freq": dataset.metadata.freq,
    "cardinality": [int(i.cardinality) for i in dataset.metadata.feat_static_cat],
    "embedding_dimension": [feature_opt["staticFeatureEmbDim"]] * len(dataset.metadata.feat_static_cat)
}


estimator = TransformerEstimator(
    **estimator_opt,
    **extra_params,
    time_features = time_features, 
    trainer_kwargs=trainer_kwargs)


#Training
start = time.time()
checkpoint_path = checkpoint_opt["startCheckpoint"] if checkpoint_opt and "startCheckpoint" in checkpoint_opt else None
checkpoint_pars = checkpoint_opt["modelCheckpoint"] if "modelCheckpoint" in checkpoint_opt else None #Parameters for ModelCheckpoint PL class

model_checkpoint = estimator.train_and_save(
    training_data=train, 
    ckpt_path=checkpoint_path, ckpt_pars=checkpoint_pars, 
    cache_data = True)

if rank == 0:
    print(f"Execution time: {time.time()-start} s")  
    print(f"Best model path: {model_checkpoint.best_model_path}")  

