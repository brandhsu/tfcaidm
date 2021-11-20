"""Tools to flatten and unflatten nested dicts"""

from tfcaidm.common.constants import DELIM


def unflatten_dict(flattened_dict, DELIM=DELIM):
    """Converts a flattened (non-nested) dict into a nested dict.

    Args:
        flattened_dict (dict): A flattened (non-nested) dict

    Returns:
        [dict]: Nested-dict.

    Example:

    flattened_dict = {
        "train/xs/dat": None,
        "train/ys/pna/mask_id": ["msk-pna"],
        "train/ys/pna/mask_weight": [1, 100],
        "train/ys/pna/metrics": ["dice"],
        "train/ys/ratio/mask_id": ["msk-ratio"],
        "train/ys/ratio/mask_weight": [0, 5],
        "train/ys/ratio/metrics": ["mse"],
        "train/trainer/seed": [0],
        "train/trainer/n_folds": [1],
        "train/trainer/batch_size": [8],
        "train/trainer/iters": [10000],
        "train/trainer/steps": [100],
        "train/trainer/valid_freq": [4],
        "train/trainer/lr": ["6e-3"],
        "train/trainer/lr_alpha": [0.25],
        "train/trainer/lr_decay": [0.97],
    }

    nested_dict = unflatten_dict(flattened_dict)

    ---

    nested_dict = {
        "train": {
            "xs": {"dat": None},
            "ys": {
                "pna": {
                    "mask_id": ["msk-pna"],
                    "mask_weight": [1, 100],
                    "metrics": ["dice"],
                },
                "ratio": {
                    "mask_id": ["msk-ratio"],
                    "mask_weight": [0, 5],
                    "metrics": ["mse"],
                },
            },
            "trainer": {
                "seed": [0],
                "n_folds": [1],
                "batch_size": [8],
                "iters": [10000],
                "steps": [100],
                "valid_freq": [4],
                "lr": ["6e-3"],
                "lr_alpha": [0.25],
                "lr_decay": [0.97],
            },
        }
    }

    """

    def insert(dct, lst):
        for x in lst[:-2]:
            dct = dct.setdefault(x, {})
        dct[lst[-2]] = lst[-1]

    nested_dict = {}

    lists = ([*k.split(DELIM), v] for k, v in flattened_dict.items())

    for lst in lists:
        insert(nested_dict, lst)

    return nested_dict


def flatten_dict(nested_dict, flattened_dict=None, parent_key=None, DELIM=DELIM):
    """Convert a nested dict into a flattened dict.

    Args:
        nested_dict (dict): A nested dict.
        flattened_dict (dict): Mutated during recursion, value should remain None.
        parent_key (str): Parent's key name, value should remain None.

    Returns:
        [dict]: Flattened dict.

    Example:

    nested_dict = {
        "train": {
            "xs": {"dat": None},
            "ys": {
                "pna": {
                    "mask_id": ["msk-pna"],
                    "mask_weight": [1, 100],
                    "metrics": ["dice"],
                },
                "ratio": {
                    "mask_id": ["msk-ratio"],
                    "mask_weight": [0, 5],
                    "metrics": ["mse"],
                },
            },
            "trainer": {
                "seed": [0],
                "n_folds": [1],
                "batch_size": [8],
                "iters": [10000],
                "steps": [100],
                "valid_freq": [4],
                "lr": ["6e-3"],
                "lr_alpha": [0.25],
                "lr_decay": [0.97],
            },
        }
    }

    flattened_dict = flatten_dict(nested_dict)

    ---

    flattened_dict = {
        "train/xs/dat": None,
        "train/ys/pna/mask_id": ["msk-pna"],
        "train/ys/pna/mask_weight": [1, 100],
        "train/ys/pna/metrics": ["dice"],
        "train/ys/ratio/mask_id": ["msk-ratio"],
        "train/ys/ratio/mask_weight": [0, 5],
        "train/ys/ratio/metrics": ["mse"],
        "train/trainer/seed": [0],
        "train/trainer/n_folds": [1],
        "train/trainer/batch_size": [8],
        "train/trainer/iters": [10000],
        "train/trainer/steps": [100],
        "train/trainer/valid_freq": [4],
        "train/trainer/lr": ["6e-3"],
        "train/trainer/lr_alpha": [0.25],
        "train/trainer/lr_decay": [0.97],
    }

    """

    if flattened_dict is None:
        flattened_dict = {}

    for k, v in nested_dict.items():
        k = f"{parent_key}{DELIM}{k}" if parent_key else k
        if isinstance(v, dict):
            flatten_dict(nested_dict=v, flattened_dict=flattened_dict, parent_key=k)
            continue

        flattened_dict[k] = v

    return flattened_dict
