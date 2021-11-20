"""Pyramid scene parsing network block"""

import math

import tfcaidm.models.layers.transform as transform
from tfcaidm.models.utils import select as model_select


def pool_block(x, c, k, s, hyperparams):
    c = int(c / 2)

    # --- Average pool
    a = transform.average_pool(x, s)
    a = model_select.conv_selection(x=a, c=c, k=k, hyperparams=hyperparams)

    # --- Max pool
    m = transform.max_pool(x, s)
    m = model_select.conv_selection(x=m, c=c, k=k, hyperparams=hyperparams)

    return [a, m]


def upsample_block(x_, d, w, h):

    for i in range(len(x_)):

        # --- Control downsampling
        x_size = x_.shape[1:-1]  # depth, width, height
        d_ = max_downsamples(x_size[0], d, 1)
        w_ = max_downsamples(x_size[1], w, 1)
        h_ = max_downsamples(x_size[2], h, 1)

        # --- Get new feature dim
        depth = auto_stride(d, d_)
        width = auto_stride(w, w_)
        height = auto_stride(h, h_)
        s = (depth, width, height)

        # --- Upsample
        x_[i] = transform.up_sample(x_[i], s=s)

    return x_


def max_downsamples(x_dim, s_dim, n):
    """Determine amount of possible downsamples"""

    try:
        return int(math.log(x_dim, s_dim)) - n
    except:
        return 1


def auto_stride(a, b):
    return int(2 ** a / 2 ** b) if a >= b else 1


def psp(x, c, k, s, hyperparams, **kwargs):
    """Implementation of PSPNet block https://arxiv.org/abs/1612.01105"""

    n_branches = hyperparams["model"]["branches"]

    # --- Control downsampling
    x_size = x.shape[1:-1]  # depth, width, height
    d = max_downsamples(x_size[0], s[0], n_branches)
    w = max_downsamples(x_size[1], s[1], n_branches)
    h = max_downsamples(x_size[2], s[2], n_branches)

    x_list = [x]
    n_channels = int(c / n_branches) if c >= n_branches else 1

    # create parallel branches
    for i in range(n_branches):

        # --- Get new feature dim
        depth = auto_stride(d, i)
        width = auto_stride(w, i)
        height = auto_stride(h, i)
        s = (depth, width, height)

        # --- Process features
        x_ = pool_block(x, n_channels, k, s, hyperparams)
        x_ = upsample_block(x_, d, w, h)

        # --- Aggregate each upsampled feature
        for feature in x_:
            x_list.append(feature)

    # --- Combine feature maps
    x = transform.add(x_list)

    # --- Refine features
    x = model_select.conv_selection(x=x, c=c, k=1, hyperparams=hyperparams)

    return x
