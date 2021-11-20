"""Defined loss functions"""

import tfcaidm.losses.funcs.entropy as entropy
import tfcaidm.losses.funcs.distance as distance
import tfcaidm.losses.funcs.dice as dice
import tfcaidm.losses.funcs.focal as focal
import tfcaidm.losses.funcs.tversky as tversky

losses = {
    "sce": entropy.SparseCategoricalCrossentropy,
    "wce": entropy.WeightedCategoricalCrossentropy,
    "mae": distance.MeanAbsoluteError,
    "mse": distance.MeanSquaredError,
    "dice": dice.DiceLoss,
    "logcosh_dice": dice.LogCoshDiceLoss,
    "tversky": tversky.TverskyLoss,
    "focal_tversky": tversky.FocalTverskyLoss,
    "focal": focal.FocalLoss,
}
