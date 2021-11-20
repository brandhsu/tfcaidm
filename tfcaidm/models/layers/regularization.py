"""Normalization layers"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


normalization_kwargs = {
    "bnorm": layers.BatchNormalization,
    "lnorm": layers.LayerNormalization,
    "none": lambda: tf.identity,
}


def normalization(hyperparams):
    k = hyperparams["model"]["norm"]

    if k not in normalization_kwargs:
        raise ValueError(f"ERROR! Normalization layer `{k}` is not defined!")

    x = normalization_kwargs[k]

    return x


dropout = lambda x, r=0.25: layers.Dropout(rate=r)(x)
l1 = lambda l1_rate=0.1: regularizers.L1(l1=l1_rate)
l2 = lambda l2_rate=0.1: regularizers.L2(l2=l2_rate)
l1_l2 = lambda l1_rate=0.1, l2_rate=0.1: regularizers.l1_l2(l1=l1_rate, l2=l2_rate)
