"""CSPNet dense block"""

import tfcaidm.models.layers.transform as transform
from tfcaidm.models.utils import select as model_select


def csp(x, c, k, hyperparams, **kwargs):
    """Implementation of csp-net block https://arxiv.org/abs/1911.11929"""

    # --- Temporary placement
    n = hyperparams["model"]["depth"]
    b = hyperparams["model"]["bneck"]
    o = c

    # --- Split feature maps into two paths
    if x.shape[-1] > 1:
        split_channels = int(x.shape[-1] / 2)
    else:
        split_channels = None

    x1 = x[..., :split_channels]
    x2 = x[..., split_channels:]

    # --- Split channels for each path
    if c > 1:
        c = int(c / 2)

    # --- Dense block
    x2 = transform.concat(
        [x2, model_select.conv_selection(x=x2, c=c, k=k, hyperparams=hyperparams)]
    )
    for i in range(1, n):
        x2 = dense_conv(x2, i + 1, c, b, hyperparams)
        x2 = transform.concat(
            [x2, model_select.conv_selection(x=x2, c=c, k=k, hyperparams=hyperparams)]
        )

    # --- Transition layer
    x2 = transition_layer(x2, c, o, k, hyperparams)
    x = transform.concat([x1, x2])
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
