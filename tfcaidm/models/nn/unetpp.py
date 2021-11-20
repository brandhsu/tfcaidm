"""UNet++ model"""

import math
import numpy as np

import tfcaidm.models.layers.extra as extra
import tfcaidm.models.nn.utils as utils
from tfcaidm.models.layers import transform


def skip_connections(x_list, ind_outer, ind_inner, ind_left):
    """Adds all intermediate skip connections"""

    x_skip = [x_list[ind_left]]

    for i in range(ind_outer):
        if ind_inner == i:
            break
        # --- Get next skip connection neuron
        ind_left -= ind_outer + 1 - i
        x_skip.append(x_list[ind_left])

    x_skip = transform.add(x_skip)

    return x_skip


def max_downsamples(x_dim, s_dim, n):
    """Determine amount of possible downsamples"""

    try:
        return int(math.log(x_dim, s_dim))
    except:
        return n


def unetpp(x, n, c, k, s, hyperparams):
    """Implementation of unet++ model"""

    # --- Control downsampling
    x_size = x.shape[1:-1]  # depth, width, height
    d_size = np.array(
        [
            max_downsamples(x_size[0], s[0], n),
            max_downsamples(x_size[1], s[1], n),
            max_downsamples(x_size[2], s[2], n),
        ]
    )

    # --- Refine input
    x = utils.in_conv(x, c, k, hyperparams)
    x = extra.layer_name(x, "eblock_0")
    print(x.shape)

    # --- Store intermediate layers and feature map channels
    x_list = [x]
    c_list = [c]
    e_list = [x]
    d_list = []
    f_list = []

    # --- Iteratively define unet
    for i in range(n):
        x, c = utils.down_conv(x, c, k, s, hyperparams)
        x = extra.layer_name(x, f"eblock_{i + 1}")

        e_list.append(x)
        x_list.append(x)
        c_list.append(c)
        bottom = -(i + 2)
        inner_size = d_size - 1
        print(x.shape)

        # --- Skip connections and upsampling
        for j in range(i + 1):
            x_skip = skip_connections(x_list, i, j, bottom)
            x, c = utils.up_conv(x, x_skip, c, k, s, inner_size, hyperparams)

            # --- Outer most decoder output
            if i == n - 1:
                x = extra.layer_name(x, f"dblock_{j}")
                d_list.append(x)

            # --- Full-sized feature map outputs
            if i == j:
                f_list.append(x)

            inner_size += 1
            x_list.append(x)
            c_list.append(c)
            print(x.shape)

        d_size -= 1
        x = x_list[bottom]
        c = c_list[bottom]

    return {
        "encoders": e_list,
        "decoders": d_list,
        "full_res": f_list,
    }
