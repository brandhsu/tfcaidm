"""Adds positional information to features"""

import numpy as np


def get_coords(arrays, shift=0, epsilon=1e-9):
    """Get a normalized coordinate map, default [-1, 1]

    For a tensor with dim:
        - (B, D, W, H, C) => CoordMap(D, W, H)
        - (B, W, H, C) => CoordMap(W, H)
        - (B, H, C) => CoordMap(H)

    Args:
        arrays (np.array): A numpy array of dim > 1
        shift (int): An optional value to shift coordinate system
        epsilon (float): Added for numerical stability

    Returns:
        np.array: Normalized coordinate map(s)
    """

    assert (
        len(arrays.shape) > 2
    ), "ERROR! np.array.shape > 2, must be at least (B, H, C)"

    # --- Get input dims
    dim_c = 1
    dim_b = len(arrays.shape) - 1
    shape = arrays.shape[-dim_b:-dim_c]

    dim_1 = dim_c
    dim_n = len(shape)

    # --- Helpers to define the coordinate transform
    coord = lambda x: np.arange(x)
    mid = lambda x: (x - 1) / 2
    norm = lambda x, x_mid: (x - x_mid) / (x_mid + epsilon)  # [-1, 1]
    ncoord = lambda x: norm(coord(x), mid(x))

    mesh = [ncoord(shape[-i]) for i in range(dim_1, dim_n + 1)]

    # --- Get coordinate maps
    maps = np.array(np.meshgrid(*mesh)) + shift

    # --- Rearrange coordinate maps
    axes = rearrange(maps.shape)
    coord_maps = np.transpose(maps, axes=axes)

    return coord_maps


def rearrange(shape):
    """Get's axis needed to rearrange np.array

    Args:
        shape (tuple): np.array.shape

    Returns:
        tuple: axis in which to rearrange np.array
    """

    n_dim = len(shape)
    dims = []

    for i in range(n_dim - 1, 0, -1):

        if n_dim - len(dims) == 2:
            dims.append(1)
            dims.append(0)
            break

        elif n_dim - len(dims) == 3:
            dims.append(1)
            dims.append(2)
            dims.append(0)
            break

        else:
            dims.append(i)

    return dims


# --- Return coordinate maps
def add_coordinates(arrays, hyperparams, **kwargs):
    coord_maps = get_coords(arrays)
    return coord_maps
