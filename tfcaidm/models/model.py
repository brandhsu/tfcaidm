"""Model hyperparameter interface"""

from tensorflow import optimizers
from tensorflow.keras import Input, Model as TFModel, models as TFmodels

from tfcaidm.common.constants import DELIM
from tfcaidm.models.utils import select as model_select
from tfcaidm.common.reproducibility import set_determinism


class Model:
    def __init__(self, client):
        self.client = client
        self.data = client.get_inputs(Input)
        self.hyperparams = client.hyperparams

        self.loss = None
        self.metrics = None
        self.optimizer = None

        set_determinism(seed=self.hyperparams["train"]["trainer"]["seed"])

    def inputs(self, input_name=None):
        return model_select.input_selection(self.data, input_name, self.hyperparams)

    def backbone(self, features):
        return model_select.model_selection(features, self.hyperparams)

    def outputs(self, features):
        return model_select.task_selection(features, self.data, self.client)

    def assemble(self, inputs, outputs):
        return TFModel(inputs=inputs, outputs=outputs)

    def build(self, input_name=None):
        """Build entire model"""

        # --- Select model inputs
        x = self.inputs(input_name)

        # --- Select a model
        x = self.backbone(x)

        # --- Select the model task
        y = self.outputs(x)

        return self.assemble(inputs=self.data, outputs=y)

    def compile(self, model):
        """Compile model with objective functions"""

        lr = self.hyperparams["train"]["trainer"]["lr"]
        optimizer = optimizers.Adam(learning_rate=lr)

        model.compile(optimizer=optimizer)

        return model

    def create(self, input_name=None):
        model = self.build(input_name)
        model = self.compile(model)

        return model

    @staticmethod
    def transfer_weights(src_model, dest_model):
        weights = src_model.get_weights()
        dest_model.set_weights(weights)

    @staticmethod
    def get_num_params(model):
        return f"{model.count_params():,}"

    @staticmethod
    def load_model(path, compile=False, custom_objects={}):
        return TFmodels.load_model(
            filepath=path,
            compile=compile,
            custom_objects=custom_objects,
        )

    @staticmethod
    def inference_mode(model, inputs, outputs):
        inputs = inference_inputs(model, names=inputs)
        outputs = inference_outputs(model, names=outputs)

        return TFModel(inputs=inputs, outputs=outputs)


def inference_inputs(model, names=[]):
    inputs = {}

    for name in names:
        n = {k.name: k for k in model.inputs}
        x = {k: v for k, v in n.items() if name == k}

        err = f"ERROR! Input `{name}` is not defined in model inputs"
        assert x, f"{err} {list(n.keys())}"

        inputs.update(x)

    return inputs


def inference_outputs(model, names=[], contains="logits"):
    outputs = {}

    get_name = lambda s: s.split(DELIM)[0]

    for name in names:
        n = {get_name(k.name): k for k in model.outputs if contains in k.name}
        y = {k: v for k, v in n.items() if name == k}

        err = f"ERROR! Output `{name}` is not defined in model outputs"
        assert y, f"{err}  {list(n.keys())}"

        outputs.update(y)

    return outputs
