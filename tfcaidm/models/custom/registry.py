"""Interface for getting available model blocks"""

from tfcaidm.common.inherit import inherit
from tfcaidm.models.custom import _zoo


@inherit(_zoo.conv_types)
def available_convs():
    return custom_conv()


@inherit(_zoo.tran_types)
def available_trans():
    return custom_tran()


@inherit(_zoo.pool_types)
def available_pools():
    return custom_pool()


@inherit(_zoo.eblocks)
def available_encoders():
    return custom_encoder()


@inherit(_zoo.dblocks)
def available_decoders():
    return custom_decoder()


@inherit(_zoo.heads)
def available_heads():
    return custom_head()


@inherit(_zoo.task_types)
def available_tasks():
    return custom_task()


@inherit(_zoo.models)
def available_models():
    return custom_model()


# ---- Functions to overload

# --- conv_type in csv
def custom_conv():
    customs = {}
    return customs


# --- conv_type in csv
def custom_tran():
    customs = {}
    return customs


# ---pool_type in csv
def custom_pool():
    customs = {}
    return customs


# --- eblock in csv
def custom_encoder():
    customs = {}
    return customs


# --- dblock in csv
def custom_decoder():
    customs = {}
    return customs


# --- head in csv
def custom_head():
    customs = {}
    return customs


def custom_task():
    customs = {}
    return customs


# --- model in csv
def custom_model():
    customs = {}
    return customs


# --- example model hyperparams
def sample_params():
    return {
        "model": {
            "model": "unet",
            "conv_type": "conv",
            "pool_type": "conv",
            "eblock": "conv",
            "elayer": 1,
            "dblock": "conv",
            "depth": 4,
            "width": 32,
            "width_scaling": 1,
            "kernel_size": [1, 3, 3],
            "strides": [1, 2, 2],
            "bneck": 2,
            "branches": 4,
            "atrous_rate": 6,
            "order": "rnc",
            "norm": "bnorm",
            "activ": "leaky",
            "attn_msk": "softmax",
        }
    }
