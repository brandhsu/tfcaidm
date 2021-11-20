"""Convolutional block attention (spatial + channel attention)"""

from tensorflow.keras import layers

import tensorflow as tf
import tfcaidm.models.blocks.eca_block as eca_block
import tfcaidm.models.blocks.se_block as se_block
import tfcaidm.models.layers.activation as activation
import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform


def channel_attention(x, c, r, hyperparams, **kwargs):
    """Implementation of channel-attention module"""

    attention = {
        "cbam_se": se_block.se,
        "cbam_eca": eca_block.eca,
    }

    eblock = hyperparams["model"]["eblock"]

    if eblock not in attention:
        raise ValueError(f"ERROR! Model block `{eblock}` is not defined!")

    return attention[eblock](x, c, hyperparams, r=r)


def spatial_attention(x, k, hyperparams, **kwargs):
    """Implementation of spatial-attention module"""

    # --- Squeeze (channel pool)
    favg = tf.reduce_mean(x, keepdims=True, axis=-1)
    fmax = tf.reduce_max(x, keepdims=True, axis=-1)

    # --- 7x7 receptive field
    x = transform.concat([favg, fmax])

    for _ in range(3):
        x = learner.pre_activation_conv(x=x, c=1, k=k)

    return activation.attn_msk(x, hyperparams)


def cbam(x, c, k, hyperparams, r=4, **kwargs):
    """Implementation of convolutional block attention https://arxiv.org/abs/1807.06521"""

    ca = channel_attention(x, c, r, hyperparams)
    cs = spatial_attention(ca, k, hyperparams)
    return x + ca * cs
