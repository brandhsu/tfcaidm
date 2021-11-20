import setup
from jarvis.train import params
from jarvis.utils.general import gpus
from tfcaidm import Trainer

# --- Autoselect GPU (use only on caidm cluster)
gpus.autoselect()

# --- Get hyperparameters
hyperparams = params.load()

# --- Train model (dataset and model created within trainer)
trainer = Trainer(hyperparams)
results = trainer.cross_validation(save=True)
trainer.save_results(results)
