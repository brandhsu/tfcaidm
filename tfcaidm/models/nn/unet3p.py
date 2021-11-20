"""UNet3+ model"""

import math

import tfcaidm.models.layers.extra as extra
import tfcaidm.models.layers.transform as transform
import tfcaidm.models.nn.utils as utils
from tfcaidm.models.utils import select as model_select


def skip_connections(x_list, c, k, hyperparams):
    """Adds all intermediate skip connections"""

    c = round(c / hyperparams["model"]["width_scaling"])

    x_skip = []
    x_size = x_list[-1].shape[1:-1]

    for i in range(len(x_list)):
        x = x_list[i]
        y_size = x.shape[1:-1]

        # --- Calculate required strides from different scales
        s = [int(math.ceil(y / x)) for x, y in zip(x_size, y_size)]

        # --- Apply pooling over intermediate skip connections
        x = model_select.pool_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)
        x_skip.append(x)

    # --- NOTE: Can aggregate using attention, add, or concat
    x_skip = transform.add(x_skip)
    x_skip = model_select.conv_selection(x=x_skip, c=c, k=k, hyperparams=hyperparams)

    return x_skip


def max_downsamples(x_dim, s_dim, n):
    """Determine amount of possible downsamples"""

    try:
        return int(math.log(x_dim, s_dim)) - n
    except:
        return 1


def unet3p(x, n, c, k, s, hyperparams):
    """Implementation of unet 3+ (unet+++) model"""

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
    for j in range(n):
        x_skip = skip_connections(e_list[: n - j], c, k, hyperparams)
        x, c = utils.up_conv(x, x_skip, c, k, s, d_size, hyperparams)
        x = extra.layer_name(x, f"dblock_{j}")

        d_list.append(x)
        d_size = [d_i + 1 if d_i else d_i for d_i in d_size]
        print(x.shape)

    return {
        "encoders": e_list,
        "decoders": d_list,
        "full_res": [e_list[0], d_list[-1]],
    }
