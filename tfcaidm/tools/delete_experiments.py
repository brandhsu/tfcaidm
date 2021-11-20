"""Delete all runs and metadata for an experiment"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser

# --- Constants
CSV = "csvs"
LOG = "logs"
SCR = "scripts"


def verify():
    response = input("\nDelete this experiment? This is NOT REVERSIBLE! [y, n]: ")
    if response.lower() == "y":
        return True
    return False


def exists(path):
    if Path(path).exists():
        return True
    return False


def experiment_deleter(csv_path):
    """Function to delete all run data associated with an experiment
    Args:
        csv_path (str): path to a hyperparam or result csv
    """

    csv_path = Path(csv_path).resolve()
    csv_name = "_".join(csv_path.stem.split("_")[:-1])

    root = csv_path.parent.parent

    csv_path = root / CSV / f"{csv_name}*"
    log_path = root / LOG / csv_name
    scr_path = root / SCR / csv_name

    cmd = "rm -rf "
    cmd += str(csv_path) + " "
    cmd += str(log_path) + " "
    cmd += str(scr_path)

    if exists(csv_path) or exists(log_path) or exists(scr_path):
        print("\n- command: {}\n".format(cmd))

        if verify():
            os.system(cmd)


def parser(args):
    p = ArgumentParser(
        description="- remove all results, logs, checkpoints, and scripts associated with a given experiment"
    )
    p.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="- path to specific experiment ie. csv_path=exp/xr_pna/csvs/2021-11-03_19-10-43_PDT_hyper.csv",
    )
    parsed = p.parse_args(args)
    arguments = {name: getattr(parsed, name) for name in vars(parsed)}

    return arguments


def delete(csv_path, **kwargs):
    experiment_deleter(csv_path)


if __name__ == "__main__":
    args = parser(sys.argv[1:])
    delete(**args)
