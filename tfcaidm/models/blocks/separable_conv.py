"""(EXPERIMENTAL) Depthwise separable convolution blocks https://arxiv.org/abs/1610.02357v3"""

import tensorflow as tf
import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform


# --- Alternative to depthwise operating on channels first then spatials
def channel_spatial_conv(x, c, k, s, d, name, hyperparams):
    channel = learner.conv(x=x, c=1, k=1, s=1, d=d, hyperparams=hyperparams)
    spatial = learner.conv(x=channel, c=c, s=s, d=d, hyperparams=hyperparams)
    return spatial


def channel_spatial_tran(x, c, k, s, d, name, hyperparams):
    channel = learner.conv(x=x, c=1, k=1, s=1, d=d, hyperparams=hyperparams)
    spatial = learner.tran(x=channel, c=c, s=s, d=d, hyperparams=hyperparams)
    return spatial


# --- Sub-modules
def depthwise_operation(x, k, s, d, hyperparams):
    x_list = []

    for i in range(x.shape[-1]):
        fm = tf.expand_dims(x[..., i], axis=-1)
        x_list.append(learner.conv(x=fm, c=1, k=k, s=s, d=d, hyperparams=hyperparams))

    return x_list


def depthwise_upsample_operation(x, k, s, d, hyperparams):
    x_list = []

    for i in range(x.shape[-1]):
        fm = tf.expand_dims(x[..., i], axis=-1)
        x_list.append(learner.tran(x=fm, c=1, k=k, s=s, d=d, hyperparams=hyperparams))

    return x_list


def extreme_reduction(x, hyperparams):
    x = learner.conv(x=x, c=1, k=1, hyperparams=hyperparams)
    return x


def pointwise_operation(x, c, hyperparams):
    x = learner.conv(x=x, c=c, k=1, hyperparams=hyperparams)
    return x


# --- Complete-modules
def separable_conv(x, c, k, s, d, name, hyperparams):
    # --- Depthwise Operation
    x = depthwise_operation(x, k, s, d, hyperparams)
    x = transform.concat(x)

    # --- Pointwise Operation
    x = pointwise_operation(x, c, hyperparams)
    return x


def separable_tran(x, c, k, s, d, name, hyperparams):
    # --- Depthwise Operation
    x = depthwise_upsample_operation(x, k, s, d, hyperparams)
    x = transform.concat(x)

    # --- Pointwise Operation
    x = pointwise_operation(x, c, hyperparams)
    return x


def smaller_separable_conv(x, c, k, s, d, name, hyperparams):
    # --- Extreme Reduction
    x = extreme_reduction(x, hyperparams)

    # --- Depthwise Operation
    x = depthwise_operation(x, k, s, d, hyperparams)
    x = transform.concat(x)

    # --- Pointwise Operation
    x = pointwise_operation(x, c, hyperparams)
    return x


def smaller_separable_tran(x, c, k, s, d, name, hyperparams):
    # --- Extreme Reduction
    x = extreme_reduction(x, hyperparams)

    # --- Depthwise Operation
    x = depthwise_upsample_operation(x, k, s, d, hyperparams)
    x = transform.concat(x)

    # --- Pointwise Operation
    x = pointwise_operation(x, c, hyperparams)
    return x


def smallest_separable_conv(x, c, k, s, d, name, hyperparams):
    # --- Extreme Reduction
    x = extreme_reduction(x, hyperparams)

    # --- (Pooling)
    x = learner.conv(x=x, c=1, k=1, s=s, hyperparams=hyperparams)

    # --- Pointwise Operation
    x = pointwise_operation(x, c, hyperparams)
    return x


def smallest_separable_tran(x, c, k, s, d, name, hyperparams):
    # --- Extreme Reduction
    x = extreme_reduction(x, hyperparams)

    # --- Determine if need to pool
    pooling = (
        hyperparams["model"]["pool_type"] != "none"
        and hyperparams["model"]["pool_type"] != "aspp"
        and hyperparams["model"]["pool_type"] != "acsp"
        and hyperparams["model"]["pool_type"] != "wasp"
    )
    if pooling:
        x = learner.tran(x=x, c=1, k=1, hyperparams=hyperparams)
    else:
        x = learner.conv(x=x, c=1, k=1, s=s, hyperparams=hyperparams)

    # --- Pointwise Operation
    x = pointwise_operation(x, c, hyperparams)
    return x
