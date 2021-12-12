"""Tensorboard viewer for a set of experiments"""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser


def tensorboard_viewer(run_path, port):
    """Function to bring up a tensorboard for a directory of experiments
    Args:
        run_path (str): path to a directory of experiments
        port     (str): port number to run tensorboard
    """

    run_path = Path(run_path).resolve()
    exps = os.listdir(run_path)
    cmd = "tensorboard --logdir_spec="

    for exp in exps:
        if os.path.isdir(run_path / exp):
            cmd += exp + ":" + str(run_path) + "/" + exp + "/" + "logdirs" + ","

    cmd = cmd[:-1]
    cmd += f" --bind_all --port {port}"
    print("\n- command: {}\n".format(cmd))
    os.system(cmd)


def parser(args):
    p = ArgumentParser(
        description="- tensorboard viewer for all runs under a given experiment"
    )
    p.add_argument(
        "--run_path",
        type=str,
        required=True,
        help="- path to specific experiment ie. run_path=exp/xr_pna/logs/2021-11-03_19-10-43_PDT",
    )
    p.add_argument("--port", type=str, default="9000", help="- port number")
    parsed = p.parse_args(args)
    arguments = {name: getattr(parsed, name) for name in vars(parsed)}

    return arguments


def view(run_path, port, **kwargs):
    tensorboard_viewer(run_path, port)


if __name__ == "__main__":
    args = parser(sys.argv[1:])
    view(**args)
