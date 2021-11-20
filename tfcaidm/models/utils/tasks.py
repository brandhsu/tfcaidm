"""Automatically determine output shape for different model tasks"""

import tfcaidm.common.constants as constants
import tfcaidm.models.layers.learner as learner
import tfcaidm.models.layers.transform as transform
import tfcaidm.models.layers.activation as activation

from tfcaidm.losses.loss import Loss
from tfcaidm.metrics.metric import Metric
from tfcaidm.models.utils import select as model_select


def auto_task_selector(x, inputs, client):
    logits = {}

    hyperparams = client.hyperparams
    outputs = hyperparams["train"]["ys"]
    output_shapes = client.get_output_shapes()

    for name, addons in outputs.items():
        feature = model_select.head_selection(x, addons)
        logit_name = constants.get_name(name, "logits")

        if len(output_shapes[name]) == 1:
            logits[logit_name] = scaler_projection(
                x=feature,
                n_classes=addons["n_classes"],
                hyperparams=hyperparams,
                name=logit_name,
            )

        else:
            logits[logit_name] = vector_projection(
                x=feature,
                y=output_shapes[name],
                n_classes=addons["n_classes"],
                hyperparams=hyperparams,
                name=logit_name,
            )

        # --- Get inputs and outputs
        y_true = inputs[name]
        y_pred = logits[logit_name]

        # --- Get loss weights and alpha
        weights = get_weights(inputs, addons)
        alpha = hyperparams["train"]["trainer"]["lr_alpha"]

        # --- Get names
        loss_name = hyperparams["train"]["ys"][name]["loss"]
        metric_name = hyperparams["train"]["ys"][name]["metric"]

        # --- Get loss and metric
        loss = Loss.add_loss(
            y_true, y_pred, name, loss_name, alpha=alpha, weights=weights
        )
        metric = Metric.add_metric(
            y_true, y_pred, name, metric_name, alpha=alpha, weights=weights
        )

        logits.update(loss)
        logits.update(metric)

    return logits


def get_weights(inputs, addons, _id="mask_id"):
    weights = None

    for k in addons:
        if type(addons[k]) == dict:
            if weights is None:
                weights = inputs[addons[k]["name"]]
            else:
                weights += inputs[addons[k]["name"]]

    return weights


# --- Helpers


def scaler_projection(x, n_classes, hyperparams, name=None):
    x = activation.attn_msk(x, hyperparams)
    x = transform.global_average_pool(x)

    return learner.pre_activation_mlp(x=x, c=n_classes, name=name)


def vector_projection(x, y, n_classes, hyperparams, name=None):
    """Reshape for [depth, width, height]"""

    # --- Get shape of each dimension
    xd, xw, xh = x.shape[-4:-1]
    yd, yw, yh = y[:-1]

    # --- Reshape and linearly project
    s1 = reshape_dim(xd, yd)
    s2 = reshape_dim(xw, yw)
    s3 = reshape_dim(xh, yh)

    return learner.pre_activation_conv(x=x, c=n_classes, k=1, s=(s1, s2, s3), name=name)


def reshape_dim(x, y):

    # --- Identity
    if x == y:
        return int(x / y)

    # --- Downsample
    elif x > y:
        return int(x / y)

    # --- Upsample
    elif x < y:
        return int(y / x)
