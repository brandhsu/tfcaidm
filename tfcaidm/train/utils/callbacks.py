"""Model.fit compatible callbacks"""

import os
from pathlib import Path
from collections.abc import Iterable

from tensorflow.keras import callbacks
from tensorboard.plugins.hparams import api as hp


def lr_scheduler(hyperparams):
    lr_decay = hyperparams["train"]["trainer"]["lr_decay"]
    assert lr_decay >= 0 and lr_decay <= 1

    return callbacks.LearningRateScheduler(
        lambda epoch, lr: lr * lr_decay if epoch > 10 else lr
    )


def model_checkpoints(hyperparams):
    # --- Create output_dir
    log_dir = hyperparams["train"]["trainer"]["log_dir"] + "/weights/"
    os.makedirs(log_dir, exist_ok=True)

    # --- Create weight file
    path = hyperparams["model"]["model"] + "_" + "{epoch:03d}.hdf5"
    filepath = log_dir + path

    return callbacks.ModelCheckpoint(
        filepath,
        monitor="val_loss",
        verbose=True,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )


def hparams_init(hyperparams):
    # --- Create log_dir
    log_dir = hyperparams["train"]["trainer"]["log_dir"]
    log_dir += "/logdirs/" + f"run_{Path(log_dir).stem}"

    return hp.KerasCallback(log_dir, fmt(hyperparams["model"]))


def tensorboard_init(hyperparams):
    # --- Create output_dir
    log_dir = hyperparams["train"]["trainer"]["log_dir"]
    log_dir += "/logdirs/" + f"run_{Path(log_dir).stem}"

    return callbacks.TensorBoard(log_dir=log_dir)


def exit_on_nan(hyperparams):
    return callbacks.TerminateOnNaN()


# --- NOTE: Not implemented yet
def reduce_lr(hyperparams, **kwargs):
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
        **kwargs,
    )


def early_stop(hyperparams, **kwargs):
    return callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )


def fmt(hyperparams):
    return {k: fmt_iter(v) for k, v in hyperparams.items()}


def fmt_iter(item):
    if isinstance(item, Iterable):
        return str(item)
    else:
        return item
