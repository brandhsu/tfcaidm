"""Defined training callbacks"""

import tfcaidm.train.utils.callbacks as callbacks

calls = {
    "checkpoint": callbacks.model_checkpoints,
    "lr_scheduler": callbacks.lr_scheduler,
    "tensorboard": callbacks.tensorboard_init,
    "hparams": callbacks.hparams_init,
    "exit_on_nan": callbacks.exit_on_nan,
}
