"""Sets random seeds"""

import os
import random
import numpy as np
import tensorflow as tf


def set_seeds(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def set_determinism(
    seed=0,
    fast=False,
    single_thread=False,
    *args,
    **kwargs,
):
    """
    Enable 100% reproducibility on operations related to tensor and randomness.

    ref: https://suneeta-mall.github.io/2019/12/22/Reproducible-ml-tensorflow.html

    Args:
        seed (int): seed value for global randomness
        fast (bool): faster performance at the cost of determinism/reproducibility
        single_thread (bool): disable thread parallelism between independent operations
    """
    set_seeds(seed=seed)

    if fast:
        return

    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    if single_thread:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
