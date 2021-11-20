"""U^2Net block"""

import math

import tfcaidm.models.layers.activation as activation
import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform
import tfcaidm.models.layers.regularization as regularization
from tfcaidm.models.utils import select as model_select
from tfcaidm.models.nn import utils as nn_utils


def down_conv(x, conv, norm, acti, s, hyperparams):
    """Encoder module"""

    x = layer(x, conv, norm, acti, hyperparams)
    x = transform.average_pool(x=x, s=s)
    return x


def up_conv(x, x_skip, conv, norm, acti, s, d, hyperparams):
    """Decoder module"""

    x = layer(x, conv, norm, acti, hyperparams)
    s = nn_utils.handle_pooling(s, d)
    x = transform.up_sample(x=x, s=s)
    x = layer(x + x_skip, conv, norm, acti, hyperparams)
    return x


def layer(x, conv, norm, acti, hyperparams):
    """Forward pass conv"""

    if hyperparams["model"]["order"] == "rnc":
        x = acti(norm(conv(x)))
    else:
        x = norm(acti(conv(x)))

    return x


def max_downsamples(x_dim, s_dim, n):
    """Determine amount of possible downsamples"""

    try:
        return int(math.log(x_dim, s_dim)) - n
    except:
        return 1


def u2net(x, c, k, s, hyperparams, **kwargs):
    """Implementation of u2net block to achieve the behavior of U2-Net https://arxiv.org/abs/2005.09007"""

    # --- Control downsampling
    n = hyperparams["model"]["branches"]
    x_size = x.shape[1:-1]  # depth, width, height
    d_size = [
        max_downsamples(x_size[0], s[0], n),
        max_downsamples(x_size[1], s[1], n),
        max_downsamples(x_size[2], s[2], n),
    ]

    # --- Shared conv layers
    conv = learner.conv_layer(c, k)
    norm = regularization.normalization(hyperparams)()
    acti = activation.nonlinearity(hyperparams)()

    # --- Store intermediate layers
    x = model_select.conv_selection(x=x, c=c, k=1, hyperparams=hyperparams)
    x_list = [x]

    # --- Encoder network
    for i in range(n):
        x = down_conv(x, conv, norm, acti, s, hyperparams)
        x_list.append(x)

    # --- Decoder network
    for i in range(n):
        x_skip = x_list[-(i + 2)]
        x = up_conv(x, x_skip, conv, norm, acti, s, d_size, hyperparams)
        d_size = [d_i + 1 if d_i else d_i for d_i in d_size]

    del x_list

    return model_select.conv_selection(x=x, c=c, k=k, hyperparams=hyperparams)
