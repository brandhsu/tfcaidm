"""Compare all experiments for a given project"""

import os
import sys
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

# --- Constants
CSV = "csvs"


def experiment_comparison(project_path):
    """Function to compare all experiments associated with a project
    Args:
        project_path (str): path to a hyperparam or result csv
    """

    project_path = Path(project_path).resolve()
    csv_path = project_path / CSV
    csvs = os.listdir(csv_path)

    results = [
        (csv_path / f)
        for f in list(
            filter(
                lambda f: f.split("_")[-1] == "results.csv",
                csvs,
            )
        )
    ]

    try:
        return pd.concat([pd.read_csv(result) for result in results])
    except:
        return pd.DataFrame()


def parser(args):
    p = ArgumentParser(
        description="- compare all experiments associated with a project"
    )
    p.add_argument(
        "--project_path",
        type=str,
        required=True,
        help="- path to a specific project csvs ie. project_path=exp/xr_pna/",
    )
    parsed = p.parse_args(args)
    arguments = {name: getattr(parsed, name) for name in vars(parsed)}

    return arguments


def compare(project_path, **kwargs):
    return experiment_comparison(project_path)


if __name__ == "__main__":
    args = parser(sys.argv[1:])
    compare(**args)
