"""Convolutional gated recurrent cell"""

import tfcaidm.models.layers.activation as activation
import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform


def project(x, c):
    if x.shape[-1] != c:
        return learner.conv_layer(c, 1)(x)
    return x


def convgru(x, x_skip, c, k, hyperparams, s=(1, 2, 2), **kwargs):
    """Implementation of conv-gru cell https://arxiv.org/abs/1412.3555"""

    # --- Project channels
    x = project(x, c)
    x_skip = project(x_skip, c)

    wz = learner.conv_layer(c, k)  # update gate
    wr = learner.conv_layer(c, k)  # reset gate
    wh = learner.conv_layer(c, k)  # gate fusion

    pool = lambda x, s: (transform.max_pool(x, s) + transform.average_pool(x, s)) / 2
    layer = lambda x, hyperparams: activation.apply_all_activations(x, hyperparams)

    x_skip = pool(x_skip, s)

    z = layer(wz(x), hyperparams) + layer(wz(x_skip), hyperparams)
    r = layer(wr(x), hyperparams) + layer(wr(x_skip), hyperparams)
    h_ = activation.tanh(wh(x) + wh(x_skip) * r)
    h = (1 - z) * x_skip + z * h_

    # --- Upsample
    if s != (1, 1, 1):
        h = transform.up_sample(x=h, s=s)

    return h
