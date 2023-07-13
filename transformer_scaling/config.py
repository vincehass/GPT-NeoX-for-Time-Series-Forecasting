import yaml, os

def generate_config(network_opt, model_opt, ext_trainer_opt, feature_opt, checkpoints, logger, folder = "", summit = True, num_nodes = 1):
    if not summit:
        trainer_opt = {
            "max_epochs": 1,
            "accelerator": "gpu",
            "strategy": "ddp",
            "devices": 2,
            "log_every_n_steps": 1,
            "gradient_clip_val": 5
        }
    else:
        trainer_opt =  {
            "max_epochs": 1,
            "accelerator": "gpu",
            "devices": 6,
            "strategy": "ddp",
            "num_nodes": num_nodes,
            "log_every_n_steps": 1,
            "gradient_clip_val": 5
        }

    config = {
        "plTrainer": trainer_opt,
        "transformer": network_opt,
        "estimator": model_opt,
        "epoch": ext_trainer_opt,
        "checkpoints": {
            "modelCheckpoint": checkpoints,
            "#loadCheckpoint": "filename",
            "#startCheckpoint": "filename"
        },
        "featureOpt": feature_opt,
        "logger": logger
    }

    with open(os.path.join(folder, "config.yaml"), 'w') as file:
        yaml.dump(config, file)
    


def read_config():
    with open("config.yaml", 'r') as file:
        opt = yaml.safe_load(file)

    return opt["plTrainer"], {**opt["estimator"], **opt["transformer"], **opt["epoch"]}, opt["featureOpt"], opt["checkpoints"], opt["logger"]
    


if __name__ == "__main__":

    network_opt = {
        "nhead": 2,
        "num_encoder_layers": 124,
        "num_decoder_layers": 6,
        "dim_feedforward": 32,
        "activation": "gelu",
        "encoder_checkpoint_step": 1,
        "decoder_checkpoint_step": 1,
    }

    model_opt = {
        "prediction_length": 24,
        "context_length": 240,
        "lags_seq": [1,2,3,4,5,6,7,24,30],
        "scaling": True
    }

    trainer_opt = {
        "batch_size": 100,
        "num_batches_per_epoch": 2
    }

    feature_opt = {
        "staticFeatureEmbDim": 3,
    }

    checkpoints = {
        "dirpath": "logs/", 
        "save_top_k": 2, 
        "monitor": "train_loss",
        "every_n_train_steps": 2,
        "verbose": True,
        "auto_insert_metric_name": False
    }
    logger = {
        "save_dir": "logs",
        "name": 'transformer_logs'
    }
    log_name = "test.log"
    generate_config(network_opt, model_opt, trainer_opt, feature_opt, checkpoints, logger, summit=False)
