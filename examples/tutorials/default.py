from __init__ import set_path

set_path()

from pprint import pprint

import tfcaidm.models.custom.registry as model_registry
import tfcaidm.losses.custom.registry as loss_registry
import tfcaidm.metrics.custom.registry as metric_registry


models = {}
models["convs"] = model_registry.available_convs()
models["trans"] = model_registry.available_trans()
models["pools"] = model_registry.available_pools()
models["encoders"] = model_registry.available_encoders()
models["decoders"] = model_registry.available_decoders()
models["heads"] = model_registry.available_heads()
models["tasks"] = model_registry.available_tasks()
models["models"] = model_registry.available_models()
pprint(models)


losses = {}
losses["losses"] = loss_registry.available_losses()
pprint(losses)


metrics = {}
metrics["metrics"] = metric_registry.available_metrics()
pprint(metrics)
