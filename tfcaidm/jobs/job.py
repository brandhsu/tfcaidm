"""Job creation for training"""

import os
import stat
import getpass
import itertools
import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML

import tfcaidm.jobs.utils.env as env
import tfcaidm.jobs.utils.config as config
import tfcaidm.common.timedate as timedate
import tfcaidm.jobs.utils.tool as tool


class Jobs(config.Config):
    def __init__(self, path, *args, **kwargs):
        super(Jobs, self).__init__(path=path, *args, **kwargs)

        self.csv_path = None
        self.log_dir = None
        self.script_dir = None
        self.scripts = None
        self.timestamp = None

    def __create_csv(self, params):
        df = pd.DataFrame(params)
        df.to_csv(self.csv_path, index=False)

    def create_folder(self, path):
        Path(path).mkdir(exist_ok=True)

    def __create_folders(self, root="exp", name="unnamed"):
        exp = Path(root).resolve()
        self.create_folder(exp)

        exp = (exp / name).resolve()
        self.create_folder(exp)

        csv_basedir = exp / "csvs"
        log_basedir = exp / "logs"
        script_basedir = exp / "scripts"

        self.create_folder(csv_basedir)
        self.create_folder(log_basedir)
        self.create_folder(script_basedir)

        self.csv_path = str(csv_basedir / self.timestamp) + "_hyper.csv"
        print(f"- csv located at {self.csv_path}")

        self.log_dir = str(log_basedir / self.timestamp)
        self.create_folder(self.log_dir)
        print(f"- logs located at {self.log_dir}")

        self.script_dir = str(script_basedir / self.timestamp)
        self.create_folder(self.script_dir)
        print(f"- scripts located at {self.script_dir}")

    def __create_log_folders(self, params):
        for i, _ in enumerate(params):
            path = str(Path(self.log_dir) / str(i))
            params[i]["env/path/param_csv"] = self.csv_path
            params[i]["train/trainer/log_dir"] = path

            self.create_folder(path)

            param = tool.unflatten_dict(params[i])

            self.create_yml(param, path, ext="pipeline.yml")

    def __create_permutations(self):
        hyperparams = []

        model_params = tool.flatten_dict(self.config["model"])
        train_params = tool.flatten_dict(self.config["train"])

        model_keys = [*model_params.keys()]
        train_keys = [*train_params.keys()]
        model_keys = [f"model/{k}" for k in model_keys]
        train_keys = [f"train/{k}" for k in train_keys]
        keys = model_keys + train_keys

        model_values = [*model_params.values()]
        train_values = [*train_params.values()]
        values = model_values + train_values
        values = [[value] if value is None else value for value in values]

        for i, row in enumerate(itertools.product(*values)):
            params = {}
            params["env/path/root"] = self.config["env"]["path"]["root"]
            params["env/path/name"] = self.config["env"]["path"]["name"]
            params["env/path/client"] = self.config["env"]["path"]["client"]

            params.update(dict(zip(keys, row)))
            hyperparams.append(params)

        return hyperparams

    def __create_scripts(
        self,
        num_gpus,
    ):
        assert num_gpus != 0, "ERROR! Training must be done using at least 1 gpu!"

        if num_gpus < 0:
            num_gpus = len(self.scripts)

        # ------------------------------------------------#
        scripts = {}
        num_models = len(self.scripts)
        leftover = num_models % num_gpus
        allocated = num_models - leftover
        models_per_gpu = allocated // num_gpus

        if allocated == 0:
            allocated = num_models
            models_per_gpu = 1

        counter = 0

        # --- Iterate over all chosen gpus
        for i in range(0, allocated, models_per_gpu):

            # --- Iterate over all models under a given gpu
            contents = ""
            for j in range(models_per_gpu):
                k = i + j
                contents += self.scripts[k]
                contents += "\n"

            scripts[counter] = contents
            counter += 1

        # --- Iterate over remaining models, if any
        if allocated < num_models:
            contents = ""
            for k in range(allocated, num_models):
                contents += self.scripts[k]
                contents += "\n"

            scripts[counter] = contents
        # ------------------------------------------------#

        # --- Populate bash script
        for k, v in scripts.items():

            # --- Write contents to a file
            path = str(Path(self.script_dir) / f"{k}.sh")
            with open(path, "w") as f:
                f.write(v)

            # --- Make script executable
            Jobs.exe(path)

    def create_yml(self, params, path, ext):
        path = str(Path(path) / ext)

        with open(path, "w") as f:
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.dump(params, f)

    def get_params(self):
        """For returning params, useful for local development and debugging"""

        params = self.__create_permutations()
        return params

    def setup(
        self,
        producer,
        consumer,
        root=None,
        name=None,
        libraries=[],
    ):
        assert (Path(producer).name != Path(consumer).name) and (
            Path(producer).resolve().parent == Path(consumer).resolve().parent
        ), f"""ERROR! Please ensure that `producer` and `consumer` are different files within the same directory!

        Issues:
            producer={Path(producer).resolve()}
            consumer={Path(consumer).resolve()}

        Note: producer must be set to __file__ and consumer must point to an external file in the same directory.
        """

        self.timestamp = timedate.get_date()

        params = self.__create_permutations()

        if root is None:
            root = self.config["env"]["path"]["root"]

        if name is None:
            name = self.config["env"]["path"]["name"]

        if (
            len(libraries) == 0
            and "lib" in self.config["env"]
            and self.config["env"]["lib"] is not None
        ):
            libraries = self.config["env"]["lib"]
            libraries = [tuple(v) for k, v in libraries.items()]

        self.__create_folders(root=root, name=name)
        self.__create_log_folders(params)
        self.__create_csv(params)
        self.__setup_scripts(params, consumer=consumer, libraries=libraries)

        return self

    def __setup_scripts(
        self,
        params,
        consumer,
        libraries=[],
    ):
        kwargs = {}
        kwargs["user"] = getpass.getuser()
        kwargs["python_path"] = os.getcwd()
        kwargs["param_path"] = self.csv_path
        kwargs["program"] = str(Path(os.getcwd()) / consumer)
        kwargs["libraries"] = libraries

        scripts = {}

        for i, param in enumerate(params):
            kwargs["log_dir"] = str(Path(self.log_dir) / str(i))
            kwargs["row_id"] = i

            command = env.set_requirements(**kwargs)
            scripts[i] = command

        print(f"- training {i + 1} model configurations")

        self.scripts = scripts

    def train_cluster(self, gpu="titan|rtx", num_gpus=1, run=True):
        """For cluster training (must be invoked in a standalone file)

        Args:
            gpu (str): Name of gpu to run, based on regex matching.
            num_gpus (int): Number of gpus to use for training.
                - Special: a value of -1 means to train all models in parallel.
                           Please be responsible when doing so!
        """

        self.__create_scripts(num_gpus=num_gpus)
        path = str(Path(self.script_dir) / "*.sh")

        if run:
            cmd = (
                "jarvis cluster add -scripts "
                + f'"{path}"'
                + f' -workers "({gpu}).*worker-[0,1]"'
            )
            os.system(cmd)
            print(f"- script {cmd} is running on the clusters.")

    def train_local(self, run=True):
        """For local training training (must be invoked in a standalone file)"""

        self.__create_scripts(num_gpus=1)
        path = str(Path(self.script_dir) / "*.sh")

        if run:
            cmd = f"sh {path}"
            os.system(cmd)
            print(f"- script {cmd} is running locally.")

    @staticmethod
    def exe(path):
        os.chmod(path, stat.S_IEXEC | stat.S_IREAD)
