"""Dataset hyperparameter interface"""

from tfcaidm.data.jclient import JClient
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
