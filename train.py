import os 
import logging
from datetime import datetime
import pytz

import torch 
from torch import nn
from torchvision import models
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import mlflow
from mlflow.tracking import MlflowClient

from utils.boq_utils import check_directory_empty
from models.boq_model import BoQModel
# from models.backbones import ResNet
from models.backbones import Backbone
from models.boq import BoQ

from dataloaders.boq_dataloader import get_train_dataloader
from dataloaders.boq_dataloader import get_val_dataloader

from utils.boq_utils import load_yaml_config
from utils.boq_utils import export_model_to_pte
from utils.boq_utils import convert_to_coreml

from torchinfo import summary

# Define IST timezone
ist = pytz.timezone('Asia/Kolkata')

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BoQ_train")
logger.setLevel(os.environ["LOGGING_LEVEL"])
console_handler = logging.StreamHandler()
formatter = logging.Formatter(os.environ["LOGGING_FORMATTER"])
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

CONFIG_PATH = os.environ["BOQ_CONFIG_PATH"]

def main():
    logger.info("Loading configuration...")
    config = load_yaml_config(CONFIG_PATH)

    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")

    backbone_name = config["model"]["backbone"]
    backbone = Backbone(
        backbone_name=backbone_name,
        pretrained=True,
        unfreeze_n_blocks=config["model"].get("unfreeze_n_blocks", 1),
        crop_last_block=config["model"].get("crop_last_block", True)
    )

    aggregator_cnf = config["model"]["aggregator"]
    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=aggregator_cnf["channel_proj"],
        num_queries=aggregator_cnf["num_queries"],
        num_layers=aggregator_cnf["num_layers"],
        row_dim=aggregator_cnf["row_dim"],
        linear_dims = aggregator_cnf["linear_dims"]
    )

    resume_ckpt_path = config["train"].get("resume_ckpt_path", None)
    model_config = config["model"]
    if resume_ckpt_path:
        logger.info(f"Resuming training from checkpoint: {resume_ckpt_path}")
        model = BoQModel.load_from_checkpoint(checkpoint_path=resume_ckpt_path, model_config=model_config, backbone=backbone, aggregator=aggregator)
    else:
        logger.info("Initializing new model...")
        model = BoQModel(model_config=model_config, backbone=backbone, aggregator=aggregator)

    print(summary(model, (1,3,224,224), device=device))
    logger.info("Preparing training dataloader...")
    train_dataloader = get_train_dataloader(dataloader_config=config["dataloaders"]["train"])
    val_dataloader = get_val_dataloader(dataloader_config=config["dataloaders"]["val"])

    logger.info("Setting up TensorBoard logger...")
    logger_tb = TensorBoardLogger(
        config["logging"]["tensorboard_log_dir"],
        name="BOQ",
        log_graph=config["logging"]["tensorboard_log_graph"]
    )

    now_ist = datetime.now(ist)
    timestamp_str = now_ist.strftime("%Y-%m-%d_time_%H:%M:%S")
    callbacks = None
    if config["train"]["callback_enabled"]:
        logger.info("Setting up model checkpoint callback...")
        model_ckpt_call_back = config["train"]["model_checkpoint_callback"]
        check_directory_empty(model_ckpt_call_back["model_ckpt_dir"])
        callbacks = [ModelCheckpoint(
            monitor=model_ckpt_call_back["monitor_param"],
            dirpath=model_ckpt_call_back['model_ckpt_dir'],
            filename=model_ckpt_call_back["model_ckpt_filename"],
            save_top_k=model_ckpt_call_back["top_k_model"],
            save_last=model_ckpt_call_back["save_last"],
            mode=model_ckpt_call_back["mode"],
            verbose=model_ckpt_call_back["verbose"]
        )]

    if config["logging"]["mlflow"]["enabled"]:
        logger.info("Initializing MLflow logging...")
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = config["logging"]["mlflow"]["system_metrics_enable"]
        mlflow.set_tracking_uri(config["logging"]["mlflow"]["tracking_uri"])
        client = MlflowClient()
        experiment_name = config["logging"]["mlflow"]["experiment_name"]
        experiment_id = client.create_experiment(experiment_name) if client.get_experiment_by_name(experiment_name) is None else client.get_experiment_by_name(experiment_name).experiment_id

    logger.info("Setting up trainer...")
    trainer = L.Trainer(
        max_epochs=config['train']['max_epochs'],
        accelerator=config['train']['accelerator'],
        log_every_n_steps=config['train']['log_every_n_steps'],
        logger=logger_tb,
        default_root_dir=config['logging']['tensorboard_log_dir'],
        check_val_every_n_epoch=config['train']["check_val_every_n_epoch"],
        callbacks=callbacks,
        precision=config['train']['precision'],
        # strategy="ddp_find_unused_parameters_false"
    )

    export_sample_input_size = config["train"]["model_export"]["transform"]
    coreml_path = config["train"]["model_export"]["coreml_path"]
    pte_path = config["train"]["model_export"]["pte_model_path"]

    if config["logging"]["mlflow"]["enabled"]:
        mlflow.pytorch.autolog(log_models=True, log_datasets=False)
        with mlflow.start_run(experiment_id=experiment_id, run_name=config["logging"]["mlflow"]["run_name"]) as run:
            logger.info("Starting training with MLflow...")
            mlflow.log_artifact(CONFIG_PATH)
            trainer.fit(model, train_dataloader, val_dataloader)
            logger.info("Training complete. Saving checkpoint...")
            trainer.save_checkpoint(f"{config['train']['save_checkpoint']}_{timestamp_str}")
            mlflow.log_artifact(config['train']['save_checkpoint'])

            example_input = torch.randn(1, 3, export_sample_input_size["input_size"], export_sample_input_size["input_size"])
            logger.info("Exporting model to PTE...")
            pte_path = export_model_to_pte(model,example_input, pte_path)

            logger.info("Exporting model to CoreML...")
            convert_to_coreml(model, example_input, coreml_path)

            mlflow.log_artifact(coreml_path)
            mlflow.log_artifact(pte_path)
        mlflow.end_run()
    else:
        logger.info("Starting training without MLflow...")
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        logger.info("Training complete. Saving checkpoint...")
        trainer.save_checkpoint(f"{config['train']['save_checkpoint']}_{timestamp_str}")
        example_input = torch.randn(1, 3, export_sample_input_size["input_size"], export_sample_input_size["input_size"])
        logger.info("Exporting model to PTE...")
        export_model_to_pte(model, example_input, pte_path)
        logger.info("Exporting model to CoreML...")
        convert_to_coreml(model, example_input, coreml_path)
    
    
if __name__ == "__main__":
    main()
