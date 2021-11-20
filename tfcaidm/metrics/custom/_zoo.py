"""Defined metrics"""

import tfcaidm.metrics.funcs.acc as acc
import tfcaidm.metrics.funcs.dice as dice
import tfcaidm.metrics.funcs.distance as distance

metrics = {
    "acc": acc.Accuracy,
    "bacc": acc.BalancedAccuracy,
    "mae": distance.MeanAbsoluteError,
    "mse": distance.MeanSquaredError,
    "dice": dice.DiceScore,
}
