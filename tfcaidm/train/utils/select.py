"""Selects training callbacks to use"""

import tfcaidm.train.custom.registry as registry

# --- callbacks in csv
def callback_selection(hyperparams):
    """get training callback functions"""

    callbacks = registry.available_callbacks()

    # --- Extract hyperparams from params csv
    cbacks = hyperparams["train"]["trainer"]["callbacks"]

    return [callbacks[k] for k in cbacks if k in callbacks]
