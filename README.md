<div align="center">
    <img src="https://raw.githubusercontent.com/Brandhsu/tfcaidm/master/docs/images/tensor.png" height="240" width="284" alt="TFCAIDM Tensor">
    <h4>TensorFlow CAIDM</h4>
    Deep learning pipeline for medical imaging
    <p align="center">
        <a href="https://brandhsu.github.io/tfcaidm-site/">Site</a> | 
        <a href="https://github.com/Brandhsu/tfcaidm/blob/master/docs/slides/tfcaidm.pdf">Slides</a> | 
        <a href="https://github.com/Brandhsu/tfcaidm/blob/master/LICENSE">License</a>
    </p>
    <img src="https://badgen.net/pypi/v/tfcaidm">
    <img src="https://badgen.net/pypi/python/tfcaidm">
    <img src="https://badgen.net/github/license/brandhsu/tfcaidm">
    <img src="https://badgen.net/badge/code%20style/black?color=black">

</div>

---

## Introduction

<strong>[TFCAIDM](https://pypi.org/project/tfcaidm/)</strong> is a unified framework for building and training medical imaging deep learning models built on top of [TensorFlow](https://www.tensorflow.org/) and [JarvisMD](https://pypi.org/project/jarvis-md/). The library supports interfacing custom datasets with `jarvis`, model development with `tensorflow`, and built-in reproducibility, traceability, and performance logging for all experiments. User's can train or extend pre-existing models that have been implemented in `MODEL_ZOO.md` or define their own.

<details>

<summary>Available Features</summary>

- [Reusable state-of-the-art deep learning model blocks](https://github.com/Brandhsu/tfcaidm-pkg/blob/main/docs/tfcaidm/models/MODEL.md)
- Support for training multiple models in parallel
- High-level interface for customizing datasets, models, loss functions, training routines, etc.
- Reproducibility, performance logging, model checkpointing, and hyperparameter tracking
</details>

<details>

<summary>Upcoming Features</summary>

- AutoML / efficient hyperparameter search
- Distributed data and model training
- Vision transformer models
- Better documentation

</details>

<details>

<summary>More Information</summary>

- YAML configuration files
- Hyperparameter tuning
- Supported models
- Customizability
- Viewing results
- Benchmarks (coming soon)

</details>

<br>

Disclaimer: The library is primarily built for users with access to the caidm clusters, though general users are also supported.

---

## Installation

The current library is supported on python 3.7 and tensorflow 2.5+, and the installation instructions provided below assume that your system is already equipped with cuda and nvcc.

<details>
<summary>Local Installation</summary>

Install using the [conda](https://www.anaconda.com/products/individual) virtual environment.

Where `user` is your account username.

```sh
user $ conda create --name tfcaidm python=3.7
user $ conda activate tfcaidm
user (tfcaidm) $ pip install tensorflow
user (tfcaidm) $ pip install jarvis-md
user (tfcaidm) $ pip install tfcaidm
```

</details>

---

## Example

Training a set of models require two separate python scripts: a training submission script and a training routine script.

<details>
<summary>Training Submission</summary>

```python
from jarvis.utils.general import gpus
from tfcaidm import Jobs

# --- Define paths
YML_CONFIG = "pipeline.yml"
TRAIN_ROUTINE_PATH = "main.py"

# --- Submit a training job
Jobs(path=YML_CONFIG).setup(
    producer=__file__,
    consumer=TRAIN_ROUTINE_PATH,
).train_cluster()
```

</details>

<details>
<summary>Automated Training Routine</summary>

```python
from jarvis.train import params
from jarvis.utils.general import gpus
from tfcaidm import Trainer

# --- Autoselect GPU (use only on caidm cluster)
gpus.autoselect()

# --- Get hyperparameters (args passed by environment variables)
hyperparams = params.load()

# --- Train model (dataset and model created within trainer)
trainer = Trainer(hyperparams)
results = trainer.cross_validation(save=True)
trainer.save_results(results)
```

</details>

<details>
<summary>Custom Training Routine</summary>

```python
from jarvis.train import params
from jarvis.utils.general import gpus, overload
from tfcaidm import JClient
from tfcaidm import Model
from tfcaidm import Trainer

# --- Autoselect GPU (use only on caidm cluster)
gpus.autoselect()

# --- Get hyperparameters (args passed by environment variables)
hyperparams = params.load()

# --- Setup custom dataset generator (more details in notebooks)
@overload(JClient)
def create_generator(self, gen_data):
    for xs, ys in gen_data:

        # --- User defined code
        xs = DataAugment(xs)

        yield xs, ys

# --- Setup custom model (more details in notebooks)
@overload(Model)
def create(self):

    # --- User defined code
    model = ViT(...)
    model.compile(...)

    return model

# --- Train model (dataset and model created within trainer)
trainer = Trainer(hyperparams)
results = trainer.cross_validation(save=True)
trainer.save_results(results)

# See notebooks for a breakdown on customizability
```

</details>

<br>

For an example project, see [examples/projects](https://github.com/Brandhsu/tfcaidm/tree/master/examples/projects). For a more detailed walkthrough of the library, see [notebooks](https://github.com/Brandhsu/tfcaidm/tree/master/notebooks).

---

## Sister Repositories

- [peterchang77/caidm](https://github.com/peterchang77/caidm)
- [peterchang77/dl_tutor](https://github.com/peterchang77/dl_tutor)
