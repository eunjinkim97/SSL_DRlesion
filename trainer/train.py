from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from optuna.integration.pytorch_lightning import \
    PyTorchLightningPruningCallback
from utils import instantiate_hydra as InstHydra


def pipeline_train(cfg, trial, pruning_monitor) -> Tuple[dict, dict]:

    # Set Random Seed
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)

    # Init Modules (Data Module, Lightning Module, Logger, Callbacks)
    dm = hydra.utils.instantiate(cfg.data_module)
    lm = hydra.utils.instantiate(cfg.lightning_module)

    callbacks = InstHydra.instantiate_callbacks(cfg.get('callbacks'))
    # callbacks += [OptunaCallback(trial, monitoring_metric=pruning_monitor, report_best_only=True)]
    logger = InstHydra.instantiate_loggers(cfg.get('logger'))

    # Init Trainer
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": dm,
        "model": lm,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # Logging Hyperparameters
    if logger:
        trainer.logger.log_hyperparams(cfg)

    # Training
    try:
        trainer.fit(lm, datamodule=dm)
    except Exception as e:  # pruned 되면 해당 except으로 빠지는 이슈
        print(e)
        return None, None

    # PyTorchLightningPruningCallback pruning
    pruning_callback = [cb for cb in callbacks if isinstance(cb, PyTorchLightningPruningCallback)]
    if pruning_callback:
        pruning_callback[0].check_pruned()

    train_metrics = trainer.callback_metrics

    # Testing
    if cfg.get('test') and trainer.is_global_zero:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = None

        # ensure testing only takes place on a single device
        cfg.trainer.devices = 1
        trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
        trainer.test(model=lm, datamodule=dm, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base='1.1', config_path='../configs', config_name='train.yaml')
def run(cfg: DictConfig) -> None:

    # Set Random Seed
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)

    # Init Modules (Data Module, Lightning Module, Logger, Callbacks)
    dm = hydra.utils.instantiate(cfg.data_module)
    lm = hydra.utils.instantiate(cfg.lightning_module)
    callbacks = InstHydra.instantiate_callbacks(cfg.get('callbacks'))
    logger = InstHydra.instantiate_loggers(cfg.get('logger'))

    # Init Trainer
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Logging Hyperparameters
    if logger:
        trainer.logger.log_hyperparams(cfg)

    ## Training
    ckpt_path = cfg.get('ckpt_path') 
    trainer.fit(lm, datamodule=dm, ckpt_path=ckpt_path)
    
    ## Testing
    if cfg.get('test') and trainer.is_global_zero:
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = None

        # ensure testing only takes place on a single device
        cfg.trainer.devices = 1
        trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
        trainer.test(model=lm, datamodule=dm, ckpt_path=ckpt_path)
    return

if __name__ == '__main__':
    run()
