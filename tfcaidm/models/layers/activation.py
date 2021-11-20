"""Nonlinear activation functions"""

import tensorflow as tf
from tensorflow.keras import activations, layers

from tfcaidm.models.layers.regularization import normalization


activation_kwargs = {
    "elu": layers.ELU,
    "relu": layers.ReLU,
    "prelu": layers.PReLU,
    "leaky": layers.LeakyReLU,
    "gelu": lambda: tf.keras.activations.gelu
    # more: https://www.tensorflow.org/api_docs/python/tf/keras/activations
}


def nonlinearity(hyperparams):
    k = hyperparams["model"]["activ"]

    if k not in activation_kwargs:
        raise ValueError(f"ERROR! Activation function `{k}` is not defined!")

    return activation_kwargs[k]


def apply_all_activations(x, hyperparams):
    if hyperparams["model"]["order"] == "rnc":  # relu(norm(conv))
        x = normalization(hyperparams)()(x)
        x = nonlinearity(hyperparams)()(x)
    else:
        x = nonlinearity(hyperparams)()(x)
        x = normalization(hyperparams)()(x)

    return x


# --- Output or attention activations
tanh = lambda x: activations.tanh(x)
sigmoid = lambda x: activations.sigmoid(x)
softmax = lambda x: activations.softmax(x)


def attn_msk(x, hyperparams):
    if hyperparams["model"]["attn_msk"] == "sigmoid":
        return sigmoid(x)

    elif hyperparams["model"]["attn_msk"] == "tanh":
        return (tanh(x) + 1) / 2

    else:
        return softmax(x)
