"""Interface for getting available metrics"""

from tfcaidm.common.inherit import inherit
from tfcaidm.metrics.custom import _zoo


@inherit(_zoo.metrics)
def available_metrics():
    return custom_metric()


# --- metric in csv
def custom_metric():
    customs = {}
    return customs
