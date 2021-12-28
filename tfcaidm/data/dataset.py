"""Dataset hyperparameter interface"""

from tfcaidm.data.jclient import JClient
from tfcaidm.jobs.utils.config import Config
from tfcaidm.jobs.utils.params import HyperParameters


class Dataset(HyperParameters):
    def __init__(self, hyperparams):
        super(Dataset, self).__init__(hyperparams=hyperparams)

    def get_client(self, fold):
        path = self.hyperparams["env"]["path"]["client"]

        configs = {
            "batch": {
                "size": self.hyperparams["train"]["trainer"]["batch_size"],
                "fold": fold,
            },
        }

        client = JClient(path, hyperparams=self.hyperparams, configs=configs)

        return client

    @classmethod
    def from_yaml(cls, path, fold):
        hyperparams = Config.load_yaml(path)
        self = cls(hyperparams)
        return self.get_client(fold)
