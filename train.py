from relea import models
from relea import data
from relea import trainers
from relea.callbacks import metrics as metrics_module
from relea.callbacks import (
    ModelCheckpoint,
    LoggingCallback
)
from relea.utils.general_utils import apply_global_seed

from pathlib import Path

import argparse
import torch
import wandb
import yaml

def main(cfg):    
    apply_global_seed(cfg["seed"])
    datamodule = getattr(data, cfg["data"]["name"])(
        **cfg["data"]["args"]
    )
    datamodule.prepare_data()
    datamodule.setup()

    project_name = cfg["project"]
    config = cfg

    # tags = ["full-finetune", "baseline", "full-train", "full-val"]
    tags = cfg["tags"]
    logger = wandb.init(project=project_name, config=config, tags=tags)

    callbacks = [
        getattr(metrics_module, f"{cfg['models']['name']}MetricsCallback")(),
        ModelCheckpoint(
            dir_path=cfg["trainer"]["args"]["checkpoint_dir"], 
            track=cfg["trainer"]["args"]["track"]
        ),
        ModelCheckpoint(
            dir_path=cfg["trainer"]["args"]["checkpoint_dir"],
            track=cfg["trainer"]["args"]["track"],
            every_epoch=True,
        ),
        LoggingCallback(logger)
    ]

    model = getattr(models, cfg["models"]["name"]).from_config(cfg)
    optimizer = getattr(torch.optim, cfg["trainer"]["args"]["optimizer"]["name"])(
        model.parameters(),
        **cfg["trainer"]["args"]["optimizer"]["args"],
    )
    trainer = getattr(trainers, cfg["trainer"]["name"])(
        accelerator=cfg["trainer"]["args"]["device"],
        max_epochs=cfg["trainer"]["args"]["max_epochs"],
        callbacks=callbacks,
        enable_checkpointing=True,
        checkpoint_dir=cfg["trainer"]["args"]["checkpoint_dir"],
        clip_grad=cfg["trainer"]["args"]["clip_grad"],
        sample_epoch=cfg["trainer"]["args"]["sample_epoch"]
    )

    train_dataloader, test_dataloader = (
        datamodule.train_dataloader(),
        datamodule.test_dataloader(),
    )

    savepath: Path = Path("./work_dirs") / cfg["models"]["name"]
    savepath.mkdir(parents=True, exist_ok=True)

    trainer.train(
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        savepath=savepath
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")

    opt = parser.parse_args()

    with open(opt.config) as stream:
        cfg = yaml.safe_load(stream)

    main(cfg)