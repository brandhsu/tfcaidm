"""Distance related loss functions"""

from tensorflow import losses
from tensorflow.keras import layers


class MeanAbsoluteError(layers.Layer):
    """Computes the mean absolute error (L1) loss"""

    def __init__(self, name="MeanAbsoluteError", **kwargs):
        super(MeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.loss = losses.MeanAbsoluteError()
        self.loss_name = name

    def call(self, y_true, y_pred, weights=None, add_loss=True, **kwargs):
        loss = self.loss(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=weights,
        )

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss


class MeanSquaredError(layers.Layer):
    """Computes the mean squared error (L2) loss"""

    def __init__(self, name="MeanSquaredError", **kwargs):
        super(MeanSquaredError, self).__init__(name=name, **kwargs)
        self.loss = losses.MeanSquaredError()
        self.loss_name = name

    def call(self, y_true, y_pred, weights=None, add_loss=True, **kwargs):
        loss = self.loss(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=weights,
        )

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss
