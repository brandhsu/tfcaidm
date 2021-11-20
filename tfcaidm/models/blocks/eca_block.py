"""Efficient channel attention block"""

import math

import tfcaidm.models.layers.activation as activation
import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform


def adaptive_kernel_size(c, scale=2, shift=1):
    """
    High-dimensional channels = longer range interactions
    Low-dimensional channels = shorter range interactions
    """

    t = int(abs((math.log(c, 2) + shift) / scale))
    return t if t % 2 else t + 1


def eca(x, c, hyperparams, **kwargs):
    """Implementation of efficient channel attention (ECA) block https://arxiv.org/abs/1910.03151"""

    # --- Squeeze (pool and combine feature maps)
    gmp = transform.global_max_pool(x, keepdims=True)
    gap = transform.global_average_pool(x, keepdims=True)

    # --- Adaptive kernel size
    c = gmp.shape[-1]
    k = (1, 1, adaptive_kernel_size(c))

    # --- Excite (convolution over squeezed features)
    conv = learner.conv_layer(c=c, k=k)
    gpl = conv(gmp) + conv(gap)

    # --- Attention
    scale = activation.attn_msk(gpl, hyperparams)

    return x * scale
