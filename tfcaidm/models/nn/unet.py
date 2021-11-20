"""UNet model"""

import math

import tfcaidm.models.layers.extra as extra
import tfcaidm.models.nn.utils as utils


def max_downsamples(x_dim, s_dim, n):
    """Determine amount of possible downsamples"""

    try:
        return int(math.log(x_dim, s_dim)) - n
    except:
        return 1


def unet(x, n, c, k, s, hyperparams):
    """Implementation of unet model"""

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
        x_skip = e_list[-(i + 2)]
        x, c = utils.up_conv(x, x_skip, c, k, s, d_size, hyperparams)
        x = extra.layer_name(x, f"dblock_{i}")

        d_list.append(x)
        d_size = [d_i + 1 if d_i else d_i for d_i in d_size]
        print(x.shape)

    return {
        "encoders": e_list,
        "decoders": d_list,
        "full_res": [e_list[0], d_list[-1]],
    }
