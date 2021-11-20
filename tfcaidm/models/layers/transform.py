"""Dimensionality transformation layers"""

import tensorflow as tf
from tensorflow.keras import layers


# --- Change feature map shape in the (D, W, H) dim.


def average_pool(x, s=(1, 2, 2), **kwargs):
    x = layers.AveragePooling3D(pool_size=s, strides=s, padding="same")(x)

    return x


def max_pool(x, s=(1, 2, 2), **kwargs):
    x = layers.MaxPooling3D(pool_size=s, strides=s, padding="same")(x)

    return x


def up_sample(x, s=(1, 2, 2)):
    return layers.UpSampling3D(size=s)(x)


def global_max_pool(x, keepdims=False):
    if tf.__version__ >= "2.6":
        return layers.GlobalMaxPooling3D(keepdims=keepdims)(x)
    else:
        pool = layers.GlobalMaxPooling3D()(x)
        if keepdims:
            return layers.Reshape((1, 1, 1, x.shape[-1]))(pool)
        return pool


def global_average_pool(x, keepdims=False):
    if tf.__version__ >= "2.6":
        return layers.GlobalAveragePooling3D(keepdims=keepdims)(x)
    else:
        pool = layers.GlobalAveragePooling3D()(x)
        if keepdims:
            return layers.Reshape((1, 1, 1, x.shape[-1]))(pool)
        return pool


# --- Change feature map shape in the (C) dim.


def broadcast(func, *args, **kwargs):
    if len(*args) > 1:
        return func(**kwargs)(*args)
    else:
        return args[0][0]


add = lambda *args, **kwargs: broadcast(tf.keras.layers.Add, *args, **kwargs)
avg = lambda *args, **kwargs: broadcast(tf.keras.layers.Average, *args, **kwargs)
concat = lambda *args, **kwargs: broadcast(tf.keras.layers.Concatenate, *args, **kwargs)


# --- Reorganize dimensions


def expand(in_feature, out_feature):
    i = len(in_feature.shape)
    o = len(out_feature.shape)
    for _ in range(i, o):
        in_feature = tf.expand_dims(in_feature, axis=-2)
    return in_feature


flatten = lambda x: layers.Flatten()(x)
