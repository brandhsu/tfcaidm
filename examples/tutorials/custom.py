from pprint import pprint
from jarvis.utils.general import overload

##############################################################
from tfcaidm.models.custom import registry as model_registry

pprint(model_registry.available_convs())


@overload(model_registry)
def custom_conv():
    return {"New Model": "TF Model"}


pprint(model_registry.available_convs())
print()

###############################################################
from tfcaidm.losses.custom import registry as loss_registry

pprint(loss_registry.available_losses())


@overload(loss_registry)
def custom_loss():
    return {"New Loss Function": "TF Loss Function"}


pprint(loss_registry.available_losses())
print()

##############################################################
from tfcaidm.metrics.custom import registry as metric_registry

pprint(metric_registry.available_metrics())


@overload(metric_registry)
def custom_metric():
    return {"New Metrics Function": "TF Metrics Function"}


pprint(metric_registry.available_metrics())
print()
