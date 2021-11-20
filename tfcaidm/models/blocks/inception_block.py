"""A crazy inception block"""

# NOTE: Send a PR if you can shorten imports!

import tfcaidm.models.layers.transform as transform
from tfcaidm.models.utils import select as model_select

import tfcaidm.models.blocks.atrous_block as atrous_block
import tfcaidm.models.blocks.atrous_block as atrous_block
import tfcaidm.models.blocks.cbam_block as cbam_block
import tfcaidm.models.blocks.csp_block as csp_block
import tfcaidm.models.blocks.dense_block as dense_block
import tfcaidm.models.blocks.eca_block as eca_block
import tfcaidm.models.blocks.psp_block as psp_block
import tfcaidm.models.blocks.se_block as se_block
import tfcaidm.models.blocks.u2net_block as u2net_block


def inception(x, c, k, hyperparams, **kwargs):
    """Modification of the inception block with dimensionality reduction https://arxiv.org/abs/1409.4842"""

    x_list = [x]
    x_shared = model_select.conv_selection(x=x, c=1, k=1, hyperparams=hyperparams)

    blocks = {
        "aspp": atrous_block.aspp,
        "cbam": cbam_block.cbam,
        "csp": csp_block.csp,
        "dense": dense_block.dense,
        "eca": eca_block.eca,
        "psp": psp_block.psp,
        "se": se_block.se,
        "u2net": u2net_block.u2net,
    }

    # --- Inception network over models/blocks/*
    for k in blocks:
        x_list.append(blocks[k](x_shared, c, k, hyperparams))
    x_list.append(
        model_select.conv_selection(x=x_shared, c=c, k=k, hyperparams=hyperparams)
    )

    # --- Combine feature maps
    x = transform.add(x_list)

    # --- Refine features
    x = model_select.conv_selection(x=x, c=c, k=1, hyperparams=hyperparams)

    return x
