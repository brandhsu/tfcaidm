"""Selects training callbacks to use"""

import tfcaidm.train.custom.registry as registry

# --- callbacks in csv
def callback_selection(hyperparams):
    """get training callback functions"""

    callbacks = registry.available_callbacks()

    # --- Extract hyperparams from params csv
    cbacks = hyperparams["train"]["trainer"]["callbacks"]

    funcs = []

    # --- Get callback functions
    for cback in cbacks:
        if cback not in callbacks:
            raise ValueError(f"ERROR! Trainer callback `{cback}` is not defined!")

        funcs.append(callbacks[cback])

    return funcs
