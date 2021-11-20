"""AutoEncoder model"""

import math

import tfcaidm.models.layers.extra as extra
import tfcaidm.models.nn.utils as utils
from tfcaidm.models.utils import select as model_select


def max_downsamples(x_dim, s_dim, n):
    """Determine amount of possible downsamples"""

    try:
        return int(math.log(x_dim, s_dim)) - n
    except:
        return 1


def up_conv(x, c, k, s, d, hyperparams):
    """Decoder module"""

    c = round(c / hyperparams["model"]["width_scaling"])
    s = utils.handle_pooling(s, d)
    x = model_select.decoder_selection(
        x=x, x_skip=x, c=c, k=k, s=1, hyperparams=hyperparams
    )
    x = model_select.tran_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)

    return x, c


def ae(x, n, c, k, s, hyperparams):
    """Implementation of autoencoder model"""

    # --- Control downsampling
    x_size = x.shape[1:-1]  # depth, width, height
    d_size = [
        max_downsamples(x_size[0], s[0], n),
        max_downsamples(x_size[1], s[1], n),
        max_downsamples(x_size[2], s[2], n),
    ]

    # --- Refine input
    x = utils.in_conv(x, c, k, hyperparams)
    x = extra.layer_name(x, "eblock_0")
    print(x.shape)

    # --- Store intermediate layers
    e_list = [x]
    d_list = []

    # --- Encoder network
    for i in range(n):
        x, c = utils.down_conv(x, c, k, s, hyperparams)
        x = extra.layer_name(x, f"eblock_{i + 1}")

        e_list.append(x)
        print(x.shape)

    # --- Decoder network
    for i in range(n):
        x, c = up_conv(x, c, k, s, d_size, hyperparams)
        x = extra.layer_name(x, f"dblock_{i}")

        d_list.append(x)
        d_size = [d_i + 1 if d_i else d_i for d_i in d_size]
        print(x.shape)

    return {
        "encoders": e_list,
        "decoders": d_list,
    }
