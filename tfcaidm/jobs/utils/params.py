"""Format and store hyperparameters dict"""

import ast

from tfcaidm.common.constants import DELIM
from tfcaidm.jobs.utils.tool import flatten_dict, unflatten_dict


class HyperParameters:
    def __init__(self, hyperparams, *args, **kwargs):
        hyperparams = HyperParameters.read(hyperparams)
        self.__hyperparams = HyperParameters.unflatten(hyperparams)
        self.results = HyperParameters.unflatten(hyperparams)

    @property
    def hyperparams(self):
        return self.__hyperparams

    @staticmethod
    def read(hyperparams):
        params = {}

        for k, v in hyperparams.items():
            try:
                v = ast.literal_eval(v)
            except:
                pass
            params[k] = v

        return params

    @staticmethod
    def flatten(hyperparams):
        return flatten_dict(hyperparams, DELIM=DELIM)

    @staticmethod
    def unflatten(hyperparams):
        return unflatten_dict(hyperparams, DELIM=DELIM)
