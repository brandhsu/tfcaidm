"""This test case will make directories in the current dir then delete them when test is done!"""

import pytest
import time
import os
from pathlib import Path

from config import YAML_PATH
from tfcaidm import Jobs
from tfcaidm.jobs import params

ROOT = "bin"
NAME = "test"
ROOT_DIR = Path(__file__).resolve().parent / ROOT

# --- Get hyperparameters
runs = Jobs(path=YAML_PATH)
hyperparams = runs.get_params()


@pytest.mark.parametrize("param", hyperparams)
def test_hyperparams(param):
    assert type(param) == dict
    hyperparams = params.HyperParameters(param).hyperparams
    print(hyperparams)

    fields = ["env", "train", "model"]
    for k in hyperparams:
        assert k in fields


num_gpus = [1, 2, 3, -1, 99]


@pytest.mark.parametrize("num_gpus", num_gpus)
def test_job_cluster(num_gpus):
    """Cluster training

    Args:
        producer (__file__): must be set to __file__
        consumer (string): file path in the same dir as producer
        root (string): base dir of experiments
        name (string): name of experiment
        libraries (list of tuples (lib, version)): optional libs to pip install
    """

    runs = Jobs(path=YAML_PATH)

    runs.setup(
        producer=__file__,
        consumer="main.py",
        root=ROOT,
        name=NAME,
        libraries=[],
    ).train_cluster(num_gpus=num_gpus, run=False)

    time.sleep(1)

    assert runs.scripts is not None


def test_clean():
    os.system(f"rm -rf {ROOT_DIR}")
