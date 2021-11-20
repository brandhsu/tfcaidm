"""Validate and store yaml config"""

import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML


class Config:
    def __init__(self, path, *args, **kwargs):
        """Wrapper class to handle and store the pipeline configuration

        Args:
            config (str): path to a yaml file or dict
        """

        path = str(Path(path).resolve())

        try:
            config = Config.load_yaml(path)
            err = Config.error(config)
            if err:
                assert False, "ERROR! Missing parameters in yaml file!"
        except ValueError:
            print("ERROR! could not load yaml file in <Class Config>!")

        try:
            Config.validate_inputs(path, config)
        except ValueError:
            print("ERROR! Missing inputs or outputs in yaml file!")

        self.config = config

    @staticmethod
    def load_yaml(path):
        with open(path, "r") as f:
            yaml = YAML(typ="safe")
            return yaml.load(f)

    @staticmethod
    def error(config):
        issues = "env" not in config
        issues = issues or ("path" not in config["env"])
        issues = issues or ("name" not in config["env"]["path"])
        issues = issues or ("root" not in config["env"]["path"])
        issues = issues or ("client" not in config["env"]["path"])

        issues = "model" not in config
        issues = issues or ("activ" not in config["model"])
        issues = issues or ("atrous_rate" not in config["model"])
        issues = issues or ("attn_msk" not in config["model"])
        issues = issues or ("bneck" not in config["model"])
        issues = issues or ("branches" not in config["model"])
        issues = issues or ("conv_type" not in config["model"])
        issues = issues or ("dblock" not in config["model"])
        issues = issues or ("depth" not in config["model"])
        issues = issues or ("eblock" not in config["model"])
        issues = issues or ("elayer" not in config["model"])
        issues = issues or ("kernel_size" not in config["model"])
        issues = issues or ("model" not in config["model"])
        issues = issues or ("norm" not in config["model"])
        issues = issues or ("order" not in config["model"])
        issues = issues or ("pool_type" not in config["model"])
        issues = issues or ("strides" not in config["model"])
        issues = issues or ("width" not in config["model"])
        issues = issues or ("width_scaling" not in config["model"])
        issues = issues or ("pool_type" not in config["model"])

        issues = "train" not in config
        issues = issues or ("trainer" not in config["train"])
        issues = issues or ("batch_size" not in config["train"]["trainer"])
        issues = issues or ("callbacks" not in config["train"]["trainer"])
        issues = issues or ("iters" not in config["train"]["trainer"])
        issues = issues or ("lr" not in config["train"]["trainer"])
        issues = issues or ("lr_alpha" not in config["train"]["trainer"])
        issues = issues or ("lr_decay" not in config["train"]["trainer"])
        issues = issues or ("n_folds" not in config["train"]["trainer"])
        issues = issues or ("seed" not in config["train"]["trainer"])
        issues = issues or ("steps" not in config["train"]["trainer"])
        issues = issues or ("valid_freq" not in config["train"]["trainer"])

        return issues

    @staticmethod
    def validate_inputs(path, config):
        # --- Load jarvis client yaml
        client = Config.load_yaml(config["env"]["path"]["client"])

        client_specs = client["specs"]
        assert "xs" in client_specs, "ERROR! Client yaml must have a `xs` field!"
        assert "ys" in client_specs, "ERROR! Client yaml must have a `ys` field!"
        client_params = [*client_specs["xs"].keys(), *client_specs["ys"].keys()]

        train_specs = config["train"]
        assert "xs" in train_specs, "ERROR! Train yaml must have a `xs` field!"
        assert "ys" in train_specs, "ERROR! Train yaml must have a `ys` field!"

        train_params = []
        train_params += get_unique_subfields(train_specs["xs"])
        train_params += get_unique_subfields(train_specs["ys"])

        assert set(client_params) == set(
            train_params
        ), f"""ERROR! Inconsistency between yaml configuration files!
        
        Issues:
            client_params = {set(client_params)}
            train_params = {set(train_params)}

            client_yml = {config["env"]["path"]["client"]}
            train_yml = {path}

        Note: client_params and train_params must be identical"""


def get_unique_subfields(field):
    """Extract all unique subfields from a given top-level field from train yml

    Example:

    field = ys:
                pna:
                    mask:
                        name: lung
                        mask_weight = 10
                    loss = sce
                    metrics = acc
                cov:
                    mask:
                        name: thor
                        mask_weight = 100
                    loss = sce
                    metrics = acc

    returns [pna, lung, cov, thor]

    Args:
        field (type): YAML field name, ex: `xs` or `ys`

    Returns:
        list: list of subfields
    """

    params = []
    for k in field:
        params += [k]
        subfield = field[k]

        if type(subfield) == dict:
            for k in subfield:

                if type(subfield[k]) == dict:
                    for n in subfield[k]:

                        if n == "name":
                            if type(subfield[k][n]) == list:
                                params += subfield[k][n]
                            else:
                                params += [subfield[k][n]]

    return params
