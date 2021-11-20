"""This test case will validate the jarvis dataset!"""

import pytest

from config import YAML_PATH
from tfcaidm import Jobs
from tfcaidm import Dataset

# --- Get hyperparameters
runs = Jobs(path=YAML_PATH)
params = runs.get_params()


@pytest.mark.parametrize("param", params)
def test_models(param):
    # TODO: decide on what to check
    # client = Dataset(param).get_client(0)

    assert True
