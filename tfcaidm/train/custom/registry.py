"""Interface for getting available training callbacks"""

from tfcaidm.common.inherit import inherit
from tfcaidm.train.custom import _zoo


@inherit(_zoo.calls)
def available_callbacks():
    return custom_callback()


# --- callbacks in csv
def custom_callback():
    customs = {}
    return customs


# --- example trainer hyperparams
def sample_params():
    return {
        "train": {
            "trainer": {
                "seed": 0,
                "n_folds": 1,
                "batch_size": 8,
                "iters": 3000,
                "steps": 100,
                "valid_freq": 6,
                "lr": 0.0003,
                "lr_alpha": 0.25,
                "lr_decay": 0.97,
                "callbacks": None,
            }
        }
    }
