"""Handle all metric related functions"""

import tfcaidm.common.constants as constants
import tfcaidm.metrics.custom.registry as registry


class Metric:
    def __init__(self):
        pass

    def create(self):
        pass

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
        class_of_interest=1,
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
                class_of_interest=class_of_interest,
            )

        def metric(y_true, y_pred):
            return zoo(y_true, y_pred)

        return metric
