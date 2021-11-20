"""Handle all loss related functions"""

import tfcaidm.common.constants as constants
import tfcaidm.losses.custom.registry as registry


class Loss:
    def __init__(self):
        pass

    def create(self):
        pass

    @classmethod
    def add_loss(
        self, y_true, y_pred, output_name, loss_name, *, alpha=0.5, weights=None
    ):
        """Add custom losses to model"""

        name = constants.get_name(output_name, "loss")
        name = constants.get_name(name, loss_name)

        loss = self.loss_selection(
            self,
            name=name,
            func=loss_name,
            weights=weights,
            alpha=alpha,
        )(y_true=y_true, y_pred=y_pred)

        return {name: loss}

    def loss_selection(
        self,
        name,
        func,
        weights=1.0,
        alpha=0.75,
        gamma=0.75,
        class_of_interest=1,
        **kwargs,
    ):
        """Selects loss function to use"""

        def zoo(y_true, y_pred):
            """Utility to choose between different loss functions"""

            loss_fns = registry.available_losses()

            if func not in loss_fns:
                raise ValueError(f"ERROR! Loss function `{func}` is not defined!")

            loss_fn = loss_fns[func](name=name)

            return loss_fn(
                y_true=y_true,
                y_pred=y_pred,
                weights=weights,
                alpha=alpha,
                gamma=gamma,
                class_of_interest=class_of_interest,
            )

        def loss(y_true, y_pred):
            return zoo(y_true, y_pred)

        return loss
