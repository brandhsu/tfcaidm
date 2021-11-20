"""Model encoder, decoder, and pooling blocks"""

from tfcaidm.models.utils import select as model_select


def in_conv(x, c, k, hyperparams):
    x = model_select.conv_selection(x=x, c=c, k=k, hyperparams=hyperparams)

    return x


def handle_pooling(s, d=None):
    """Handle pooling size to ensure valid feature map size"""

    if d is not None:
        size = lambda s, d: 1 if d < 0 else s
        s = [size(s_i, d_i) for s_i, d_i in zip(s, d)]

    return s


def residual(x, c, k, s, hyperparams):
    """Implementation of resnet block"""

    x = model_select.conv_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)

    for _ in range(hyperparams["model"]["elayer"] - 1):
        x += model_select.encoder_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)

    return x


def down_conv(x, c, k, s, hyperparams):
    """Encoder module"""

    c = round(c * hyperparams["model"]["width_scaling"])
    s = handle_pooling(s)

    x = residual(x=x, c=c, k=k, s=1, hyperparams=hyperparams)
    x = model_select.pool_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)

    return x, c


def up_conv(x, x_skip, c, k, s, d, hyperparams):
    """Decoder module"""

    c = round(c / hyperparams["model"]["width_scaling"])
    s = handle_pooling(s, d)

    x_skip = model_select.decoder_selection(
        x=x, x_skip=x_skip, c=c, k=k, s=s, hyperparams=hyperparams
    )
    x = model_select.tran_selection(x=x, c=c, k=k, s=s, hyperparams=hyperparams)
    x = model_select.conv_selection(
        x=(x + x_skip), c=c, k=k, s=1, hyperparams=hyperparams
    )

    return x, c
