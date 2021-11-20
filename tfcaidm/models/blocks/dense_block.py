"""DenseNet block"""

import tfcaidm.models.layers.transform as transform
from tfcaidm.models.utils import select as model_select


def dense(x, c, k, hyperparams, **kwargs):
    """Implementation of dense-net block https://arxiv.org/abs/1608.06993"""

    # --- Temporary placement
    n = hyperparams["model"]["depth"]
    b = hyperparams["model"]["bneck"]
    o = c

    # --- Dense block
    x = transform.concat(
        [x, model_select.conv_selection(x=x, c=c, k=k, hyperparams=hyperparams)]
    )
    for i in range(1, n):
        x = dense_conv(x, i + 1, c, b, hyperparams)
        x = transform.concat(
            [x, model_select.conv_selection(x=x, c=c, k=k, hyperparams=hyperparams)]
        )

    # --- Transition layer
    x = transition_layer(x, c, o, k, hyperparams)

    return x


def dense_conv(x, n, c, b, hyperparams):
    """Dense conv to optionally compute bottleneck"""

    # --- Bottleneck if too many channels (num_layers > bottleneck ratio)
    if n >= b:
        x = model_select.conv_selection(x=x, c=(b * c), k=1, hyperparams=hyperparams)
    else:
        x = model_select.conv_selection(x=x, c=(n * c), k=1, hyperparams=hyperparams)

    return x


def transition_layer(x, c, o, k, hyperparams):
    """Transition layer following a dense block"""

    x = model_select.conv_selection(x=x, c=c, k=k, hyperparams=hyperparams)
    return model_select.conv_selection(x=x, c=o, k=1, hyperparams=hyperparams)
