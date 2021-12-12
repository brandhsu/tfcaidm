"""Accuracy related metrics"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import metrics


reduce = lambda x: tf.math.reduce_sum(x)


class Accuracy(layers.Layer):
    def __init__(self, name="Accuracy", **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)
        self.metric = metrics.Accuracy()

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        class_of_interest=1,
        add_metric=True,
        **kwargs,
    ):

        true = tf.cast(y_true[..., 0] == class_of_interest, tf.float32)
        pred = tf.cast(tf.math.argmax(y_pred, axis=-1) == class_of_interest, tf.float32)

        metric = self.metric(y_true=true, y_pred=pred, sample_weight=weights)

        if add_metric:
            self.add_metric(metric, name=self.metric.name)

        return metric


class BalancedAccuracy(layers.Layer):
    def __init__(self, name="BalancedAccuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
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

        tp = reduce(true * pred)
        fp = reduce((1 - true) * pred)
        fn = reduce(true * (1 - pred))
        tn = reduce((1 - true) * (1 - pred))

        sens = tp / (tp + fn + epsilon)
        spec = tn / (tn + fp + epsilon)

        A = sens + spec
        B = 2

        metric = A / B

        if add_metric:
            self.add_metric(metric, name=self.metric_name)

        return metric
