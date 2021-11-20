"""Add class weights (for use in loss func)"""

import numpy as np


def vector_weights(xs, ys, hyperparams, condition=True):

    # --- Create a mask template
    msk = np.ones(xs.shape) * (not hyperparams["remove_bg"])

    # --- Binarize masks
    crop = xs > 0
    label = ys > 0

    # --- Set class weights
    msk[crop] = hyperparams["mask_weight"]
    msk[label] = hyperparams[
        "output_weight" if "output_weight" in hyperparams else "mask_weight"
    ]

    return msk * condition


def scalar_weights(xs, ys, hyperparams, condition=True):

    # --- Force mask to 1
    xs[:] = 1 * condition

    return xs


def ignore_row(row, name="cohort-uci"):
    """
    Mask depending on available annotations
    Uci data has ratio no seg, RSNA has seg no ratio
    This is an example for the dual modality covid-biomarker dataset
    """

    if name in row:
        uci_ratio = row[name] == True
        rsna_mask = not uci_ratio
    else:
        uci_ratio = True
        rsna_mask = True

    return uci_ratio, rsna_mask


def modify_loss(xs, ys, row, hyperparams, kwargs):

    n_dim = len(xs.shape)

    uci_ratio, rsna_mask = ignore_row(row)

    # --- Ratio mask
    if n_dim == 1:
        condition = uci_ratio
        return scalar_weights(xs, ys, hyperparams, condition=condition)

    # --- Segmentation mask
    else:
        condition = rsna_mask
        return vector_weights(xs, ys, hyperparams, condition=condition)
