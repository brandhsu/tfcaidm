"""Focal related loss functions"""

import tensorflow as tf

from tfcaidm.losses.funcs.entropy import WeightedCategoricalCrossentropy

reduce = lambda x: tf.math.reduce_sum(x)


class FocalLoss(WeightedCategoricalCrossentropy):
    """Implementation of binary focal loss"""

    def __init__(self, name="FocalLoss", **kwargs):
        super(FocalLoss, self).__init__(name=name, **kwargs)
        self.loss_name = name

    def call(
        self,
        y_true,
        y_pred,
        weights=None,
        gamma=2.0,
        class_of_interest=1,
        add_loss=True,
        **kwargs,
    ):
        # --- Weighted cross-entropy loss (log-term)
        loss = WeightedCategoricalCrossentropy.call(
            self,
            y_true=y_true,
            y_pred=y_pred,
            weights=weights,
            class_of_interest=class_of_interest,
            add_loss=False,
        )

        # ---- Extract the class of interest
        y_true = tf.cast(y_true[..., 0] == class_of_interest, tf.float32)
        y_pred = tf.nn.softmax(y_pred, axis=-1)[..., class_of_interest]

        # --- Calculate modulation to pos and neg labels
        modulation_pos = (1 - y_pred) ** gamma
        modulation_neg = y_pred ** gamma

        mask = tf.dtypes.cast(y_true, dtype=tf.bool)
        modulation = tf.where(mask, modulation_pos, modulation_neg)

        loss = reduce(modulation * loss)

        if add_loss:
            self.add_loss(loss)
            self.add_metric(loss, name=self.loss_name)

        return loss
