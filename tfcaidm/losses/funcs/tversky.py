"""Tversky related loss functions"""

import tensorflow as tf
from tensorflow.keras import layers


reduce = lambda x: tf.math.reduce_sum(x)


class TverskyLoss(layers.Layer):
    """Computes the tversky (generalized dice) loss"""

    def __init__(self, name="TverskyLoss", **kwargs):
        super(TverskyLoss, self).__init__(name=name, **kwargs)
        self.loss_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        alpha=0.3,
        class_of_interest=1,
        epsilon=1e-9,
        add_loss=True,
        **kwargs,
    ):
        """
        Implementation of Tversky index loss
        TL = tp/(tp + (1-alpha)*fn + alpha*fp) st. alpha = [0, 1]
        = tp/fn when alpha = 0 (punish fn more)
        = tp/(t+p) when alpha = 0.5 (dice score)
        = tp/fp when alpha = 1 (punish fp more)
        """

        # ---- Extract the class of interest
        true = tf.cast(y_true[..., 0] == class_of_interest, tf.float32)
        pred = tf.nn.softmax(y_pred, axis=-1)[..., class_of_interest]

        if weights is not None:
            true *= tf.cast(weights[..., 0], tf.float32)
            pred *= tf.cast(weights[..., 0], tf.float32)

        tp = reduce(true * pred)
        fn = reduce(true * (1 - pred))
        fp = reduce((1 - true) * pred)

        A = tp
        B = tp + ((1 - alpha) * fn) + (alpha * fp) + epsilon

        tt = reduce(true * true)
        ft = reduce(true * (1 - true))

        C = tt
        D = tt + ft + epsilon

        loss = (C / D) - (A / B)

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss


class FocalTverskyLoss(TverskyLoss):
    """Computes the focal tversky (generalized dice) loss"""

    def __init__(self, name="FocalTverskyLoss", **kwargs):
        super(FocalTverskyLoss, self).__init__(name=name, **kwargs)
        self.loss_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        alpha=0.3,
        gamma=0.75,
        class_of_interest=1,
        epsilon=1e-9,
        add_loss=True,
        **kwargs,
    ):
        loss = (
            TverskyLoss.call(
                self,
                y_true=y_true,
                y_pred=y_pred,
                weights=weights,
                alpha=alpha,
                class_of_interest=class_of_interest,
                epsilon=epsilon,
                add_loss=False,
            )
            ** gamma
        )

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss
