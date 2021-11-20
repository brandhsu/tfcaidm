"""Atrous convolution blocks"""

import numpy as np

from tfcaidm.models.utils import select as model_select
import tfcaidm.models.layers.transform as transform


def aspp(x, c, k, s, hyperparams):
    """Atrous spatial pyramid pooling https://arxiv.org/pdf/1606.00915.pdf"""

    x = model_select.conv_selection(x=x, c=c, k=1, s=1, hyperparams=hyperparams)

    x_list = [x]
    n_branches = get_branches(hyperparams)

    for i in range(n_branches):
        d = get_dilation_size(x, hyperparams, scale=i + 1)
        x_new = model_select.conv_selection(x=x, c=c, k=k, d=d, hyperparams=hyperparams)
        x_list.append(x_new)

    x = transform.add(x_list)
    x = model_select.conv_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)

    return x


def acsp(x, c, k, s, hyperparams):
    """Atrous convolution spatial pooling https://arxiv.org/pdf/1706.05587.pdf"""

    x = model_select.conv_selection(x=x, c=c, k=1, s=1, hyperparams=hyperparams)

    n_branches = get_branches(hyperparams)

    for i in range(n_branches):
        d = get_dilation_size(x, hyperparams, scale=i + 1)
        x = model_select.conv_selection(x=x, c=c, k=k, d=d, hyperparams=hyperparams)

    x = model_select.conv_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)

    return x


def wasp(x, c, k, s, hyperparams):
    """Waterfall atrous spatial pooling https://arxiv.org/pdf/1706.05587.pdf"""

    x = model_select.conv_selection(x=x, c=c, k=1, s=1, hyperparams=hyperparams)

    x_list = [x]
    n_branches = get_branches(hyperparams)

    for i in range(n_branches):
        d = get_dilation_size(x, hyperparams, scale=i + 1)
        x = model_select.conv_selection(x=x, c=c, k=k, d=d, hyperparams=hyperparams)
        x_list.append(x)

    x = transform.add(x_list)
    x = model_select.conv_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)

    return x


def get_branches(hyperparams):
    return hyperparams["model"]["branches"]


def get_dilation_size(x, hyperparams, scale=1):
    is_3d = True if x.shape[1] > 1 else False

    atrous_rate = hyperparams["model"]["atrous_rate"]
    kernel_size = np.array(hyperparams["model"]["kernel_size"])

    new_kernel = kernel_size * atrous_rate * scale

    if not is_3d:
        return (1, *new_kernel[1:])

    return tuple(new_kernel)
