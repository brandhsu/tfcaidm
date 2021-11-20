"""Squeeze and excitation block"""

import math

import tfcaidm.models.layers.activation as activation
import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform


def se(x, c, hyperparams, r=4, **kwargs):
    """Implementation of squeeze-and-exication module https://arxiv.org/abs/1709.01507"""

    c = x.shape[-1]

    # --- Squeeze (pool and combine feature maps)
    gmp = transform.global_max_pool(x, keepdims=True)
    gap = transform.global_average_pool(x, keepdims=True)

    # --- Excite (convolution over squeezed features)
    conv1 = learner.conv_layer(c=int(math.ceil(c / r)), k=1)
    conv2 = learner.conv_layer(c=c, k=1)
    activ = activation.nonlinearity(hyperparams)
    gpl = conv2(activ(conv1(gmp))) + conv2(activ(conv1(gap)))

    # --- Attention
    scale = activation.attn_msk(gpl, hyperparams)

    return x * scale
