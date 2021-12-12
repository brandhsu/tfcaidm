"""Dice score related metrics"""

import tensorflow as tf
from tensorflow.keras import layers

from tfcaidm.common.constants import DELIM


reduce = lambda x: tf.math.reduce_sum(x)


class DiceScore(layers.Layer):
    """
    Computes the dice-score (F1) coefficient metric for a given

    Args:
        y_true (np.ndarray): ground-truth label
        y_pred (np.ndarray): predicted logits scores
        c (int) : class to calculate DSC on

    """

    def __init__(self, name="DiceScore", **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.metric_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        class_of_interest=1,
        epsilon=1e-9,
        add_metric=True,
        **kwargs,
    ):
        true = tf.cast(y_true[..., 0] == class_of_interest, tf.float32)
        pred = tf.cast(tf.math.argmax(y_pred, axis=-1) == class_of_interest, tf.float32)

        if weights is not None:
            true *= tf.cast(weights[..., 0] != 0, tf.float32)
            pred *= tf.cast(weights[..., 0] != 0, tf.float32)

        A = reduce(true * pred) * 2
        B = reduce(true) + reduce(pred) + epsilon

        metric = A / B

        if add_metric:
            self.add_metric(
                metric, name=(self.metric_name + f"{DELIM}{class_of_interest}")
            )

        return metric
