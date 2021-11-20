"""This test case will validate the creation of different models!"""

import pytest
from tensorflow.keras import Input

from config import YAML_PATH
from tfcaidm import Jobs
from tfcaidm import Model
from tfcaidm.jobs import params

# --- Get hyperparameters
runs = Jobs(path=YAML_PATH)
hyperparams = runs.get_params()
inputs = [(8, 16, 16, 3), (1, 8, 8, 1)]


class FakeClient(params.HyperParameters):
    def __init__(self, hyperparams, input_shape):
        params.HyperParameters.__init__(self, hyperparams)
        self.input_shape = input_shape

    def get_inputs(self, *args, **kwargs):
        return self.input_shape


@pytest.mark.parametrize("param", hyperparams)
@pytest.mark.parametrize("inputs", inputs)
def test_models(param, inputs):
    input_shape = Input(inputs)
    client = FakeClient(param, input_shape)
    assert Model(client).backbone(input_shape)
