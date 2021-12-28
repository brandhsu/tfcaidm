"""Model benchmarking interface (only available on caidm cluster)"""

import os
import stat
import getpass
import numpy as np
import pandas as pd
from pathlib import Path

from jarvis.train.client import Client
from tfcaidm.common import timedate
from tfcaidm.benchmarks import metadata
from tfcaidm.metrics.custom import registry


class Specs:
    def __init__(self, specs):
        self.specs = specs

    @property
    def output_name(self):
        return self.specs["output"]

    @property
    def metric_name(self):
        return self.specs["metric"]

    @property
    def metrics(self):
        metrics = registry.available_metrics()
        name = metadata.mapper(self.metric_name)
        return metrics[name]


class Benchmark:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.specs = Specs(metadata.specs(dataset))
        self.path = metadata.yaml_path(self.specs.specs)
        self.configs = {"batch": {"size": 1, "fold": metadata.FOLD}}

        self._results = []

    @property
    def results(self):
        return self._results

    @property
    def score(self):
        return self._score

    def yield_data(self):
        """Yield a sample data, target pair"""
        client = Client(self.path, configs=self.configs)
        _, valid = client.create_generators(test=True)
        xs, ys = next(valid)
        return xs, ys

    def check(self, y_true, y_pred):
        assert (
            self.specs.output_name in y_pred
        ), f"ERROR! Expecting model inference to return dictionary: {{{self.specs.output_name}: Tensor}}"

        true_shape = y_true[self.specs.output_name].shape[:-1]
        pred_shape = y_pred[self.specs.output_name].shape[:-1]

        assert (
            true_shape == pred_shape
        ), f"ERROR! Expecting output_shape={true_shape} but getting pred_shape={pred_shape}"

    def infer(self, model, xs, **kwargs):
        """To set the correct outputs during model inference.
        For expected inputs and outputs, use `Benchmark.help()`.

        NOTE: This method (only this method) can be modified (overloaded).

        Args:
            model (tf model): A trained tensorflow model
            xs (dict): The inputs ingested by the model

        Returns:
            dict : The outputs of the model
        """

        return model(xs)

    def run(self, model, **kwargs):
        results = []

        client = Client(self.path, configs=self.configs)
        _, gen_data = client.create_generators(test=True)

        for xs, ys in gen_data:

            name = model.output_names
            pred = self.infer(model, xs, **kwargs)

            if type(pred) != dict:
                pred = {k: v.numpy() for k, v in zip(name, pred)}

            self.check(ys, pred)

            y_true = ys[self.specs.output_name]
            y_pred = pred[self.specs.output_name]

            result = self._score()(y_true, y_pred)
            results.append(result)

        self._results = float(np.array(results).mean())
        self.model_name = model.name
        self.num_params = f"{model.count_params():,}"

        return {self.specs.metric_name: self._results}

    def _score(self):
        return self.specs.metrics()

    def submit(self, desc="trained with love <3"):

        # --- Submit scores
        assert (
            type(self._results) == float
        ), f"ERROR! Model results must be of type float!"

        schema = metadata.model_schema(
            user=getpass.getuser(),
            dataset=self.dataset,
            model=self.model_name,
            num_params=self.num_params,
            desc=desc,
            metric=self.specs.metric_name,
            score=self._results,
            date=timedate.get_mdy(),
        )

        schema = {k: [v] for k, v in schema.items()}
        path = Path(metadata.leaderboard(self.dataset)) / f"{getpass.getuser()}.csv"
        save_to_csv(schema, path)

        print(
            f"- congratulations your run with a(n) {self.specs.metric_name} of {self._results:.3f} has been submitted!"
        )

    @classmethod
    def leaderboard(cls, dataset=None):
        Path(metadata.leaderboard(dataset))

        # --- Parse leaderboard file structure
        path = Path(metadata.leaderboard(dataset=None))
        dirs = filter(lambda d: (path / d).is_dir(), os.listdir(path))
        dirs = [path / dir for dir in dirs]

        if dataset is not None:
            dirs = filter(lambda d: (d).stem == dataset, dirs)

        csvs = [
            filter(lambda f: Path(f).suffix == ".csv", os.listdir(dir)) for dir in dirs
        ]
        csvs = [dir / f for dir, csv in zip(dirs, csvs) for f in csv]

        # --- Convert csvs to pandas dataframes
        leaderboad = []
        leaderboad += [pd.read_csv(csv) for csv in csvs]

        # --- Sort dataframe
        if len(leaderboad):
            df = pd.concat(leaderboad)
            df["sort"] = pd.to_datetime(df["Date"], format="%b %d, %Y")
            df = df.sort_values(by="sort")
            df = df.drop(columns=["sort"])
        else:
            df = pd.DataFrame(
                [
                    metadata.model_schema(
                        user=None,
                        dataset=None,
                        model=None,
                        num_params=None,
                        desc=None,
                        metric=None,
                        score=None,
                        date=None,
                    )
                ]
            )

        print("- leaderboard is ordered by most recent submissions first.")

        return df

    @classmethod
    def help(cls, dataset=None):

        schema = metadata.data_table()

        if dataset is not None:
            schema = [schema[dataset]]

        path = metadata.CAIDM_CONFIG["datasets"]
        print(f"CAIDM Benchmark Datasets - updated {os.path.getmtime(path)}")

        return pd.DataFrame(schema)


def save_to_csv(df, path):
    if path.is_file():
        os.chmod(path, stat.S_IWRITE | stat.S_IWGRP | stat.S_IWOTH)
        pd.DataFrame(df).to_csv(path, index=False, mode="a", header=False)
    else:
        pd.DataFrame(df).to_csv(path, index=False)
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
