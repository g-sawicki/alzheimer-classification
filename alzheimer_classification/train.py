import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from alzheimer_classification.data.alzheimer_data_module import AlzheimerDataModule
from alzheimer_classification.models import get_model


def train(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg.model.name).to(device)

    # Setup data module
    train_transform = model.model.get_train_transform()
    test_transform = model.model.get_test_transform()

    data_module = AlzheimerDataModule(
        batch_size=cfg.data.batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.callbacks.model_checkpoint.monitor,
        mode=cfg.callbacks.model_checkpoint.mode,
        filename=cfg.callbacks.model_checkpoint.filename,
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.callbacks.early_stopping.monitor,
        patience=cfg.callbacks.early_stopping.patience,
        mode=cfg.callbacks.early_stopping.mode,
    )

    mlflow.pytorch.autolog()
    with mlflow.start_run(log_system_metrics=True, run_name="training_test") as run:
        mlf_logger = MLFlowLogger(
            experiment_name="alzheimer-classification",
            tracking_uri=mlflow.get_tracking_uri(),
            log_model=True,
            run_id=run.info.run_id,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.trainer.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=mlf_logger,
            log_every_n_steps=1,
            # fast_dev_run=True,
            # limit_train_batches=1,
            # limit_val_batches=1,
            # limit_test_batches=1,
            # num_sanity_val_steps=0,
        )

        trainer.fit(model, data_module)
        trainer.test(model, data_module)


@hydra.main(config_path="../configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
