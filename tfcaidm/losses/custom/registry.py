"""Interface for getting available losses"""

from tfcaidm.common.inherit import inherit
from tfcaidm.losses.custom import _zoo


@inherit(_zoo.losses)
def available_losses():
    return custom_loss()


# --- loss in csv
def custom_loss():
    customs = {}
    return customs
