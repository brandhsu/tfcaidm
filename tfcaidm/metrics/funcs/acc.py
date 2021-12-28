"""Accuracy related metrics"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import metrics

from tfcaidm.metrics.metric import MetricUtils


class Accuracy(layers.Layer):
    def __init__(self, name="Accuracy", **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.multis = metrics.Accuracy()
        self.binary = metrics.BinaryAccuracy()

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        add_metric=True,
        **kwargs,
    ):

        func = self.multis if y_pred.shape > 1 else self.binary
        metric = func(y_true=y_true, y_pred=y_pred, sample_weight=weights)

        if add_metric:
            self.add_metric(metric, name=self.metric.name)

        return metric


class BalancedAccuracy(layers.Layer):
    def __init__(self, name="BalancedAccuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.multis = metrics.Accuracy()
        self.binary = metrics.BinaryAccuracy()

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        add_metric=True,
        **kwargs,
    ):

        balanc_weight = MetricUtils.inverse_class_weights(y_true)
        sample_weight = MetricUtils.tf_f32(weights != 0) * balanc_weight

        func = self.multis if y_pred.shape > 1 else self.binary
        metric = func(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

        if add_metric:
            self.add_metric(metric, name=self.metric.name)

        return metric
