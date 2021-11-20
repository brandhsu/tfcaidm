"""Basic neural network layers"""

from tensorflow.keras import layers

from tfcaidm.models.layers.activation import apply_all_activations


def conv_layer(c, k=(1, 3, 3), s=(1, 1, 1), d=1, name=None):
    layer = layers.Conv3D(
        filters=c, kernel_size=k, strides=s, dilation_rate=d, name=name, padding="same"
    )

    return layer


def tran_layer(c, k=(1, 3, 3), s=(1, 2, 2), d=1, name=None):
    layer = layers.Conv3DTranspose(
        filters=c, kernel_size=k, strides=s, dilation_rate=d, name=name, padding="same"
    )

    return layer


def pre_activation_conv(x, c, k=(1, 3, 3), s=(1, 1, 1), d=1, name=None):
    x = conv_layer(c=c, k=k, s=s, d=d, name=name)(x)

    return x


def pre_activation_tran(x, c, k=(1, 3, 3), s=(1, 2, 2), d=1, name=None):
    x = tran_layer(c=c, k=k, s=s, d=d, name=name)(x)

    return x


def pre_activation_mlp(x, c, name=None):
    x = layers.Dense(units=c, name=name)(x)

    return x


def conv(x, c, hyperparams, k=(1, 3, 3), s=(1, 1, 1), d=1, name=None, **kwargs):
    x = pre_activation_conv(x, c, k=k, s=s, d=d, name=name)
    x = apply_all_activations(x, hyperparams)

    return x


def tran(x, c, hyperparams, k=(1, 3, 3), s=(1, 2, 2), d=1, name=None, **kwargs):
    x = pre_activation_tran(x, c, k=k, s=s, d=d, name=name)
    x = apply_all_activations(x, hyperparams)

    return x


def mlp(x, c, hyperparams, name=None, **kwargs):
    x = pre_activation_mlp(x, c, name=name)
    x = apply_all_activations(x, hyperparams)

    return x
