"""Entropy related loss functions"""

import tensorflow as tf
from tensorflow import losses
from tensorflow.keras import layers


class SparseCategoricalCrossentropy(layers.Layer):
    """Computes the categorical cross-entropy loss"""

    def __init__(self, name="SparseCategoricalCrossentropy", **kwargs):
        super(SparseCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.loss = losses.SparseCategoricalCrossentropy(from_logits=True)
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


class WeightedCategoricalCrossentropy(layers.Layer):
    """Computes the weighted cross-entropy loss"""

    def __init__(self, name="WeightedCategoricalCrossentropy", **kwargs):
        super(WeightedCategoricalCrossentropy, self).__init__(name=name, **kwargs)
        self.loss = tf.nn.weighted_cross_entropy_with_logits
        self.loss_name = name

    def call(
        self, y_true, y_pred, weights=None, class_of_interest=1, add_loss=True, **kwargs
    ):
        # ---- Extract the class of interest
        labels = tf.cast(y_true[..., 0] == class_of_interest, tf.float32)
        logits = tf.cast(y_pred[..., class_of_interest], tf.float32)

        # pos_weight > 1 decreases: false negatives, increases: false positives
        # pos_weight < 1 decreases: false positives, increases: false negatives
        weights = 1.0 if weights is None else weights
        pos_weight = tf.cast(tf.math.reduce_max(weights), tf.float32)

        loss = self.loss(
            labels=labels,
            logits=logits,
            pos_weight=pos_weight,
        )

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss
