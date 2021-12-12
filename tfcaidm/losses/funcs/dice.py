"""Dice score related loss functions"""

import tensorflow as tf
from tensorflow.keras import layers

from tfcaidm.losses.funcs.entropy import SparseCategoricalCrossentropy


reduce = lambda x: tf.math.reduce_sum(x)


class DiceLoss(layers.Layer):
    """Computes the soft dice-score (F1) loss"""

    def __init__(self, name="DiceLoss", **kwargs):
        super(DiceLoss, self).__init__(name=name, **kwargs)
        self.loss_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        class_of_interest=1,
        epsilon=1e-9,
        add_loss=True,
        **kwargs,
    ):
        # ---- Extract the class of interest
        true = tf.cast(y_true[..., 0] == class_of_interest, tf.float32)
        pred = tf.nn.softmax(y_pred, axis=-1)[..., class_of_interest]

        if weights is not None:
            true *= tf.cast(weights[..., 0], tf.float32)
            pred *= tf.cast(weights[..., 0], tf.float32)

        A = reduce(true * pred) * 2
        B = reduce(true) + reduce(pred) + epsilon

        C = reduce(true * true)
        D = reduce(true) + epsilon

        loss = (C / D) - (A / B)

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss


class LogCoshDiceLoss(DiceLoss):
    """Computes the soft logcosh dice-score (F1) loss"""

    def __init__(self, name="LogCoshDiceLoss", **kwargs):
        super(LogCoshDiceLoss, self).__init__(name=name, **kwargs)
        self.loss_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        class_of_interest=1,
        epsilon=1e-9,
        add_loss=True,
        **kwargs,
    ):
        loss = DiceLoss.call(
            self,
            y_true=y_true,
            y_pred=y_pred,
            weights=weights,
            class_of_interest=class_of_interest,
            epsilon=epsilon,
            add_loss=False,
        )
        cosh = lambda x: (tf.exp(x) + tf.exp(-x)) / 2.0

        loss = tf.math.log(cosh(loss))

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss


class DiceCrossentropyLoss(SparseCategoricalCrossentropy, DiceLoss):
    """Computes a weighted dice-score cross-entropy loss"""

    def __init__(self, name="DiceCrossentropyLoss", **kwargs):
        super(DiceCrossentropyLoss, self).__init__(name=name, **kwargs)
        self.loss_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        class_of_interest=1,
        epsilon=1e-9,
        alpha=0.5,
        add_loss=True,
        **kwargs,
    ):
        entropy = SparseCategoricalCrossentropy.call(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=weights,
            add_loss=False,
        )

        dice = DiceLoss.call(
            self,
            y_true=y_true,
            y_pred=y_pred,
            weights=weights,
            class_of_interest=class_of_interest,
            epsilon=epsilon,
            add_loss=False,
        )

        loss = alpha * dice + (1 - alpha) * entropy

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss


class MultiDiceLoss(DiceLoss):
    """Computes the soft dice-score (F1) loss over several classes"""

    def __init__(self, name="MultiDiceLoss", **kwargs):
        super(MultiDiceLoss, self).__init__(name=name, **kwargs)
        self.loss_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        classes_of_interest=[0, 1],
        epsilon=1e-9,
        add_loss=True,
        **kwargs,
    ):

        losses = []
        for class_of_interest in classes_of_interest:
            loss = DiceLoss.call(
                self,
                y_true=y_true,
                y_pred=y_pred,
                weights=(weights, None)[class_of_interest == 0],
                class_of_interest=class_of_interest,
                epsilon=epsilon,
                add_loss=False,
            )

            losses.append(loss)

        loss = sum([loss for loss in losses]) / len(losses)

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss
