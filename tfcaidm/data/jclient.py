"""Client hyperparameter interface"""

from jarvis.train.client import Client

from tfcaidm.common.reproducibility import set_determinism
from tfcaidm.data.utils import class_weights
from tfcaidm.data.utils import positional_encoding
from tfcaidm.jobs.utils.config import get_unique_subfields
from tfcaidm.jobs.utils.params import HyperParameters


# --- Wrapper for extracting client yml inputs and outputs
def get_yml_params(func):
    def extract(self, *args, **kwargs):
        tasks_dict = func(self, *args, **kwargs)
        targets = {}

        # --- Associate inputs and outputs with their shape
        for k, v in tasks_dict:
            shape = v["shape"]["saved"]
            targets[k] = shape

        return targets

    return extract


class JClient(HyperParameters, Client):
    def __init__(self, path, hyperparams, configs={}, *args, **kwargs):
        HyperParameters.__init__(self, hyperparams, *args, **kwargs)
        set_determinism(self.hyperparams["train"]["trainer"]["seed"])
        Client.__init__(
            self,
            path,
            configs=configs,
            custom_layers=True,
            *args,
            **kwargs,
        )

    @get_yml_params
    def get_shapes(self, key):
        return self.specs[key].items()

    def get_input_shapes(self):
        specs = self.get_shapes("xs")
        params = get_unique_subfields(self.hyperparams["train"]["xs"])

        xs = {}

        for param in params:
            for k in specs.keys():
                if k == param:
                    xs[param] = specs[k]

        return xs

    def get_output_shapes(self):
        specs = self.get_shapes("xs")
        params = get_unique_subfields(self.hyperparams["train"]["ys"])

        ys = {}

        for param in params:
            for k in specs.keys():
                if k == param:
                    ys[param] = specs[k]

        return ys

    def dataset_size(self, fold=0):
        df = self.db.header
        train = sum(df["valid"] != fold)
        valid = sum(df["valid"] == fold)

        return {"train": f"{train:,}", "valid": f"{valid:,}"}

    def apply_preprocess(self, xs, ys, row, key, params, **kwargs):
        """Applies additional feature preprocessing

        Currently supported:
            - (OUTPUT) Class loss weights
            - (INPUT) Coordinate localization map

        Args:
            xs (np.array): model array inputs
            ys (np.array): model array targets
            row (str): name of data sample (defined in client.db.header)
            key (str): name of specific input or output (defined in train.yml)
            params (dict): hyperparameter dict

        Returns:
            np.array or none: Returns new features if possible else none
        """

        if key == "coord":
            return positional_encoding.add_coordinates(xs, params, kwargs)
        elif key == "mask":
            return class_weights.modify_loss(xs, ys, row, params, kwargs)

        return

    def assign_preprocess(self, xs, ys, hyperparams, row, **kwargs):
        """Assigns preprocessing functions depending on train.yml config

        Args:
            xs (np.array): model array inputs
            ys (np.array): model array targets
            hyperparams (dict): hyperparameter dict
            row (str): name of data sample (defined in client.db.header)
        """

        for entry in hyperparams:
            params = hyperparams[entry]
            if params is not None:

                for key in params:
                    if type(params[key]) == dict:
                        name = params[key]["name"]

                        if entry in ys:
                            feature = self.apply_preprocess(  # outputs
                                xs[name],
                                ys[entry],
                                row,
                                key,
                                params[key],
                                **kwargs,
                            )
                        else:
                            feature = self.apply_preprocess(  # inputs
                                xs[name],
                                ys[entry],
                                row,
                                key,
                                params[key],
                                **kwargs,
                            )

                        if feature is not None:
                            xs[name] = feature

    def preprocess(self, arrays, row, **kwargs):

        # --- Extract input / output fields
        inputs = self.hyperparams["train"]["xs"]
        outputs = self.hyperparams["train"]["ys"]

        # --- Get `ys` from `xs`
        xs = arrays["xs"]
        ys = {k: xs[k] for k in get_unique_subfields(outputs)}

        # --- Add additional features i.e. feature map coordinates
        self.assign_preprocess(xs, ys, inputs, row, **kwargs)

        # --- Set loss weights
        self.assign_preprocess(xs, ys, outputs, row, **kwargs)

        return arrays

    def create_generator(self, gen_data):
        for xs, ys in gen_data:
            yield xs, ys

    def create_generators(self, arr=None, test=False, **kwargs):
        if arr is None:
            gen_train, gen_valid = Client.create_generators(self, test=test, **kwargs)
            gen_train = self.create_generator(gen_train)
            gen_valid = self.create_generator(gen_valid)
            yield from (gen_train, gen_valid)

        else:
            while not test:
                for xs, ys in arr:
                    yield xs, ys

            for xs, ys in arr:
                yield xs, ys
