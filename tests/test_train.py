"""This test case will validate that the model can train!"""

import pytest
import numpy as np
from tensorflow.keras import Input

from config import YAML_PATH
from tfcaidm import Jobs
from tfcaidm import Model
from tfcaidm import Trainer
from tfcaidm.jobs import params

# NOTE: This test is hardcoded!
INPUT_SHAPE = [1, 32, 32, 1]
OUTPUT_SHAPE = [1, 32, 32, 1]


# --- Get hyperparameters
runs = Jobs(path=YAML_PATH)
hyperparams = runs.get_params()


class FakeClient(params.HyperParameters):
    def __init__(self, hyperparams, input_shape):
        params.HyperParameters.__init__(self, hyperparams)
        self.input_shape = input_shape

    def get_inputs(self, *args, **kwargs):
        return self.input_shape

    def get_output_shapes(self):
        return {"lbl": OUTPUT_SHAPE, "msk": OUTPUT_SHAPE}

    def generator(self, batch_size=1):
        while True:
            xs = []
            ys = []

            for _ in range(batch_size):
                arrays = {}
                arrays["dat"] = np.zeros(INPUT_SHAPE)
                arrays["msk"] = np.zeros(OUTPUT_SHAPE)
                arrays["lbl"] = np.zeros(OUTPUT_SHAPE)
                xs.append(arrays)

            keys = ["dat", "msk", "lbl"]
            xs = {k: np.stack([x[k] for x in xs]) for k in keys}
            ys = {}

            yield xs, ys

    def create_generators(self):
        gen_train = self.generator()
        gen_valid = self.generator()
        return gen_train, gen_valid


@pytest.mark.parametrize("param", hyperparams)
def test_models(param):
    input_shape = {
        "dat": Input(INPUT_SHAPE),
        "msk": Input(OUTPUT_SHAPE),
        "lbl": Input(OUTPUT_SHAPE),
    }
    client = FakeClient(param, input_shape)

    gen_train, gen_valid = client.create_generators()
    model = Model(client).create()

    assert Trainer(param).fit(
        model,
        gen_train,
        gen_valid,
        iters=2,
        steps_per_epoch=1,
        validation_freq=2,
        callbacks=None,
    )
