"""Setup python environment"""


import os
import pytz
import getpass
import inspect
from datetime import datetime


def get_pst_time():
    """Function to get PDT time and date in 12hr format"""
    date_format = "%m_%d_%Y_%H_%M_%S_%Z"
    date_format = "%b %d %Y %-I:%M %p"
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone("US/Pacific"))
    pstDateTime = date.strftime(date_format)
    return pstDateTime


def setup(user, param_path, row_id, python_path, log_dir, **kwargs):
    """Function to setup environment

    Args:
        user        (str): account username
        param_path  (str): hyper-parameter csv path
        row_id      (int): rid in the param_path
        python_path (str): python path
        log_dir     (str): experiment output directory
    """

    env = f"""
        echo "{get_pst_time()}" > {log_dir}/stdout

        export JARVIS_PATH_CONFIGS=/home/{user}/.jarvis
        export JARVIS_PARAMS_CSV={param_path}
        export JARVIS_PARAMS_ROW={row_id}
        export PYTHONPATH={python_path}
    """

    return inspect.cleandoc(env)


def install(log_dir, libraries, **kwargs):
    """Function to pip install needed requirements

    Args:
        libraries (list): list of tuples containing library and version
        log_dir   (str): experiment output directory
    """

    inst = ""

    for library in libraries:
        lib, ver = library
        inst += f"{verify(lib, ver)} >> {log_dir}/stdout"
        inst += "\n"

    return inspect.cleandoc(inst)


def verify(library, version, **kwargs):
    """Function to check if install already exists

    Args:
        library (str): name of library to install
        version (str): version of library
    """

    ver = f"""
        ver=$(pip list | grep {library} | head -1 | xargs)
        if [ "$ver" != "{library} {version}" ]
        then
            pip install {library}=={version}
        fi
    """

    return ver


def run(program, log_dir, **kwargs):
    """Function to run python file

    Args:
        program (str): program filename to run
        log_dir (str): experiment output directory
    """

    exe = f"""
        python {program} >> {log_dir}/stdout 2>&1
    """

    return inspect.cleandoc(exe)


def set_requirements(**kwargs):
    """Function to setup all requirements and environment variables"""

    setup_sequence = []
    setup_sequence.append(setup(**kwargs))
    setup_sequence.append(install(**kwargs))
    setup_sequence.append(run(**kwargs))

    command = ""

    for step in setup_sequence:
        command += step
        command += "\n"

    return command


if __name__ == "__main__":
    # --- Example arguments
    kwargs = {}

    kwargs["user"] = getpass.getuser()
    kwargs["param_path"] = os.getcwd()
    kwargs["python_path"] = os.getcwd()
    kwargs["log_dir"] = os.getcwd()
    kwargs["program"] = os.getcwd() + "/" + "main.py"
    kwargs["row_id"] = 0

    kwargs["libraries"] = [
        ("tensorflow", "2.5.0"),
        ("pytorch", "1.9.0"),
        ("torchvision", "0.10.0"),
        ("pytorch-lightning", "1.4.2"),
    ]

    command = set_requirements(**kwargs)
    print(command)
