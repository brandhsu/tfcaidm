"""Handle all metric related functions"""

import tensorflow as tf

import tfcaidm.common.constants as constants
import tfcaidm.metrics.custom.registry as registry


class Metric:
    @classmethod
    def add_metric(
        self, y_true, y_pred, output_name, metric_name, *, alpha=0.5, weights=None
    ):
        """Add custom metrics to model"""

        name = constants.get_name(output_name, "metric")
        name = constants.get_name(name, metric_name)

        metric = self.metric_selection(
            self,
            name=name,
            func=metric_name,
            weights=weights,
            alpha=alpha,
        )(y_true=y_true, y_pred=y_pred)

        return {name: metric}

    def metric_selection(
        self,
        name,
        func,
        weights=None,
        **kwargs,
    ):
        """Selects metric function to use"""

        def zoo(y_true, y_pred):
            """Utility to choose between different metric functions"""

            metrics = registry.available_metrics()

            if func not in metrics:
                raise ValueError(f"ERROR! Metric function `{func}` is not defined!")

            metric = metrics[func](name=name)

            return metric(
                y_true=y_true,
                y_pred=y_pred,
                weights=weights,
            )

        def metric(y_true, y_pred):
            return zoo(y_true, y_pred)

        return metric


# --- Helper
class MetricUtils:
    argmax = lambda x: tf.math.argmax(x, axis=-1)
    sum = lambda x: tf.math.reduce_sum(x)
    mean = lambda x: tf.math.reduce_mean(x)
    tf_f32 = lambda x: tf.cast(x, tf.float32)
    squeeze = lambda x: tf.squeeze(x)

    @classmethod
    def get_true(cls, y_true, class_of_interest):
        true = cls.tf_f32(y_true)
        true = true == class_of_interest
        return cls.squeeze(true)

    @classmethod
    def get_pred(cls, y_pred, class_of_interest, threshold=0.5):
        pred = cls.tf_f32(y_pred)
        pred = (
            cls.argmax(pred) == class_of_interest
            if pred.shape > 1
            else cls.tf_f32(pred > threshold)
        )
        return cls.squeeze(pred)

    @classmethod
    def get_unique(cls, y_true):
        true = tf.reshape(y_true, [-1])
        classes, _ = tf.unique(true)
        for c in classes:
            yield c

    @classmethod
    def inverse_class_weights(cls, y_true):
        cls_counts = tf.math.bincount(y_true)
        cls_freqs = tf.math.reciprocal_no_nan(cls.tf_f32(cls_counts))
        weights = tf.gather(cls_freqs, y_true)
        return weights
