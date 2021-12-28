"""Dice score related metrics"""

import tensorflow as tf
from tensorflow.keras import layers

from tfcaidm.common import constants
from tfcaidm.metrics.metric import MetricUtils


class DiceScore(layers.Layer):
    """Computes the dice-score (F1) coefficient metric over all classes"""

    def __init__(self, name="DiceScore", **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.metric_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        add_metric=True,
        **kwargs,
    ):

        metric = {}
        classes = MetricUtils.get_unique(y_true)

        # --- Dice score per class
        for class_of_interest in classes:
            name = constants.get_name(self.metric_name, class_of_interest)
            metric[name] = self.compute(y_true, y_pred, class_of_interest, weights)

            if add_metric:
                self.add_metric(metric[name], name=name)

        # --- Macro-averaged dice score
        name = constants.get_name(self.metric_name, "avg")
        metric[name] = MetricUtils.mean([*metric.values()])

        if add_metric:
            self.add_metric(metric[name], name=name)

        return metric

    def compute(
        self,
        y_true,
        y_pred,
        class_of_interest,
        weights=None,
        epsilon=1e-9,
        threshold=0.5,
    ):
        """
        Computes the dice-score (F1) coefficient metric for a given class

        Args:
            y_true (tf.Tensor): ground-truth label
            y_pred (tf.Tensor): predicted logits scores
            class_of_interest (int) : class to calculate dice-score on

        """

        true = MetricUtils.get_true(y_true, class_of_interest)
        pred = MetricUtils.get_pred(y_pred, class_of_interest, threshold)

        if weights is not None:
            true *= MetricUtils.tf_f32(MetricUtils.squeeze(weights != 0))
            pred *= MetricUtils.tf_f32(MetricUtils.squeeze(weights != 0))

        A = MetricUtils.sum(true * pred) * 2
        B = MetricUtils.sum(true + pred) + epsilon

        metric = A / B

        return metric
