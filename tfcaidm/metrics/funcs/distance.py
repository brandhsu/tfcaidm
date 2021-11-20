"""Distance related metrics"""

from tensorflow.keras import layers
from tensorflow.keras import metrics


class MeanAbsoluteError(layers.Layer):
    """Computes the mean absolute error (L1) metric"""

    def __init__(self, name="MeanAbsoluteError", **kwargs):
        super(MeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.metric = metrics.MeanAbsoluteError()

    def call(self, y_true, y_pred, weights=None, add_metric=True, **kwargs):

        metric = self.metric(y_true=y_true, y_pred=y_pred, sample_weight=weights)

        if add_metric:
            self.add_metric(metric, name=self.metric.name)

        return metric


class MeanSquaredError(layers.Layer):
    """Computes the mean squared error (L2) metric"""

    def __init__(self, name="MeanSquaredError", **kwargs):
        super(MeanSquaredError, self).__init__(name=name, **kwargs)
        self.metric = metrics.MeanSquaredError()

    def call(self, y_true, y_pred, weights=None, add_metric=True, **kwargs):

        metric = self.metric(y_true=y_true, y_pred=y_pred, sample_weight=weights)

        if add_metric:
            self.add_metric(metric, name=self.metric.name)

        return metric
