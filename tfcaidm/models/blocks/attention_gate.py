"""Attention-gate block"""

import tfcaidm.models.layers.activation as activation
import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform
from tfcaidm.models.utils import select as model_select


def attention_gate(x, x_skip, hyperparams, s=(1, 2, 2), **kwargs):
    """Implementation of attention-gate block https://arxiv.org/abs/1804.03999)"""

    c = x.shape[-1]

    # --- Gating signal
    gs = model_select.conv_selection(x=x, c=c, k=1, hyperparams=hyperparams)

    # --- Skip signal
    ss = model_select.conv_selection(x=x_skip, c=c, k=1, s=s, hyperparams=hyperparams)

    # --- Combine signals
    cs = activation.nonlinearity(hyperparams)()(gs + ss)

    # --- Attention
    a = learner.pre_activation_conv(x=cs, c=1, k=1)
    a = activation.attn_msk(a, hyperparams)

    # --- Upsample
    if s != (1, 1, 1):
        a = transform.up_sample(x=a, s=s)

    return x_skip * a
