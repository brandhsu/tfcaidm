"""Benchmark related metadata"""

from pathlib import Path
from tfcaidm.jobs.utils.config import Config


# --- Paths
CAIDM_CONFIG = {
    "datasets": "/data/benchmarks/datasets.yml",
    "leaderboards": "/data/benchmarks/leaderboards",
}
FOLD = 0
NB_URL = (
    lambda proj_id: f"https://colab.research.google.com/github/peterchang77/caidm/blob/master/datasets/{proj_id}/segmentation.ipynb"
)


# --- Helpers
def load():
    try:
        return Config.load_yaml(CAIDM_CONFIG["datasets"])
    except:
        raise OSError("ERROR! Benchmarks are only available on the caidm clusters!")


def yaml_path(specs):
    return specs["code"] + specs["yaml"]


def specs(dataset):
    datasets = load()

    if dataset not in datasets:
        raise ValueError("ERROR! Dataset {dataset} is not supported!")

    return datasets[dataset]


def leaderboard(dataset):
    path = Path(CAIDM_CONFIG["leaderboards"])

    if dataset is not None:
        specs(dataset)
        return path / dataset

    return path


def mapper(name):
    return {
        "accuracy": "acc",
        "dice score": "dice",
        "mean absolute error": "mae",
    }[name]


# --- Tables
def data_schema(
    dataset,
    client,
    desc,
    demo="n/a",
    fold=0,
    **kwargs,
):
    return {
        "Dataset": dataset,
        "Client": client,
        "Description": desc,
        "Demo": demo,
        "Fold": fold,
    }


def model_schema(
    user,
    dataset,
    model,
    num_params,
    desc,
    metric,
    score,
    date,
    **kwargs,
):
    return {
        "User": user,
        "Dataset": dataset,
        "Model": model,
        "# Params": num_params,
        "Description": desc,
        "Metric": metric,
        "Score": score,
        "Date": date,
    }


def data_table():
    datasets = load()

    schema = []

    for k, v in datasets.items():
        data = data_schema(k, yaml_path(v), v["task"])
        schema.append(data)

    return schema
