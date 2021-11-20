"""Training hyperparameter interface"""

import numpy as np
from pathlib import Path


import tfcaidm.common.timedate as timedate
from tfcaidm.models.model import Model
from tfcaidm.data.dataset import Dataset
from tfcaidm.jobs.utils.params import HyperParameters
from tfcaidm.train.utils.select import callback_selection
from tfcaidm.train.utils.summary import Summary, save_results


class Trainer(HyperParameters):
    def __init__(self, hyperparams):
        HyperParameters.__init__(self, hyperparams)
        self.timestamp = timedate.get_date()

    def cross_validation(self, n_folds=None, save=False, callbacks=[]):
        """K-fold cross validation (up to K=5)

        Args:
            n_folds (int): Number of cross validation folds. Defaults to hyperparams["train"]["trainer"]["n_folds"]
            save (bool): Save client and model after model training. Defaults to False
            callbacks (List, None):
                - if None, disable callbacks
                - if non-empty, use callbacks from list
                - if emtpy list, use callbacks from hyperparams["train"]["trainer"]["callbacks"]

        Returns:
            dict: Returns a dictionary containing outputs from model.fit and model.eval
        """

        n_folds = choose(n_folds, self.results["train"]["trainer"]["n_folds"])
        assert n_folds > 0, "ERROR! n_folds must be greater than 0!"

        histories = {
            "train_history": [],
            "train_results": [],
            "valid_results": [],
        }

        # --- Cross validation generator
        cv = self.__cross_valid(n_folds, save, callbacks)

        for history in cv:
            hist = history
            train_history, train_results, valid_results = hist

            # --- Accumulate results
            histories["train_history"].append(train_history.history)
            histories["train_results"].append(train_results)
            histories["valid_results"].append(valid_results)

        return histories

    def __cross_valid(self, n_folds, save, callbacks):
        # --- Cross validation loop
        for fold in range(n_folds):

            client = Dataset(self.hyperparams).get_client(fold)

            # --- Load train dataset
            gen_train, gen_valid = client.create_generators(test=False)

            # --- Load validation dataset
            gen_train_test, gen_valid_test = client.create_generators(test=True)

            # --- Create a model
            model = Model(client).create()

            # --- Train and evaluate model
            train_history = self.fit(model, gen_train, gen_valid, callbacks=callbacks)
            train_results = self.eval(model, gen_train_test)
            valid_results = self.eval(model, gen_valid_test)

            if save:
                self.__checkpoint(client, model, fold)

            history = [train_history, train_results, valid_results]

            yield history

    def __checkpoint(self, client, model, fold=0):
        output_dir = self.hyperparams["train"]["trainer"]["log_dir"]
        client_name = Path(self.hyperparams["env"]["path"]["client"]).stem
        model_name = self.hyperparams["model"]["model"]

        client_path = output_dir + "/" + client_name + f"_{fold}" + ".yml"
        model_path = output_dir + "/" + model_name + f"_{fold}"

        self.save_client(client, client_path)
        self.save_model(model, model_path)

    def eval(self, model, gen_data, verbose=0):
        results = model.evaluate(
            x=gen_data,
            batch_size=1,
            verbose=verbose,
            return_dict=True,
        )

        return results

    def fit(
        self,
        model,
        gen_train,
        gen_valid,
        *,
        iters=None,
        steps_per_epoch=None,
        validation_freq=None,
        callbacks=[],
    ):
        """Model training

        Args:
            model (TFModel): A compiled tensorflow model
            gen_train (generator): Training dataset generator
            gen_valid (generator): Validation dataset generator
            iters (int): Total number of training iterations. Defaults to hyperparams["train"]["trainer"]["iters"]
            steps_per_epoch (int): Number of forward passes per epoch. Defaults to hyperparams["train"]["trainer"]["steps"]
            validation_freq (int): Run model validation every N epochs. Defaults to hyperparams["train"]["trainer"]["valid_freq"]
            callbacks (List, None):
                - if None, disable callbacks
                - if non-empty, use callbacks from list
                - if emtpy list, use callbacks from hyperparams["train"]["trainer"]["callbacks"]

        Returns:
            model.history: A model.history object
        """

        assert (iters is None and steps_per_epoch is None) or (
            iters is not None and steps_per_epoch is not None
        ), "ERROR! Must define both or neither `iters` and `steps_per_epoch`..."

        iters = choose(iters, self.results["train"]["trainer"]["iters"])
        steps_per_epoch = choose(
            steps_per_epoch, self.results["train"]["trainer"]["steps"]
        )
        validation_freq = choose(
            validation_freq, self.results["train"]["trainer"]["valid_freq"]
        )

        epochs = int(iters / steps_per_epoch)

        if callbacks is not None and len(callbacks) == 0:
            funcs = callback_selection(self.hyperparams)
            callbacks = [func(self.hyperparams) for func in funcs]

        history = model.fit(
            x=gen_train,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=gen_valid,
            validation_steps=steps_per_epoch,
            validation_freq=validation_freq,
            callbacks=callbacks,
        )

        # --- Save number of model params
        self.results["model"]["num_params"] = Model.get_num_params(model)

        return history

    def save_client(self, client, path):
        client.to_yml(path)

    def save_model(self, model, path):
        model.save(path)

    def save_inference(self, model, gen_train, gen_valid, output_dir, fold):
        train_output = output_dir + "/" + "train" + f"_{fold}" + ".npz"
        valid_output = output_dir + "/" + "valid" + f"_{fold}" + ".npz"

        self.save_outputs(model, gen_train, path=train_output)
        self.save_outputs(model, gen_valid, path=valid_output)

    def save_outputs(self, model, gen_data, path):
        array = lambda list_: np.array(list_ if len(list_) else [0])

        data = {
            "xs": [],
            "ys": [],
            "zs": [],
        }

        for xs, ys in gen_data:
            zs = model.predict(xs)
            data["xs"].append(xs)
            data["ys"].append(ys)
            data["zs"].append(zs)

        data = {k: array(v) for k, v in data.items()}
        np.savez_compressed(path, **data)

    def save_results(self, histories, name="val_loss"):
        assert (
            type(histories) == dict
        ), "ERROR! Argument `histories` must be of type dict!"

        valid_freq = self.results["train"]["trainer"]["valid_freq"]

        summary = Summary(histories["train_history"])
        results = {"history": summary.summarize_best_history(valid_freq, name)}
        self.update_results(results)

        summary = Summary(histories["train_results"])
        results = {"train_eval": summary.summarize_eval()}
        self.update_results(results)

        summary = Summary(histories["valid_results"])
        results = {"valid_eval": summary.summarize_eval()}
        self.update_results(results)

        start_time = self.timestamp
        end_time = timedate.get_date()

        self.results["train"]["trainer"]["train_time"] = timedate.timediff(
            end_time, start_time
        )

        save_results(self.flatten(self.results))

    def update_results(self, new_results):
        new_results = self.flatten(new_results)
        results = self.flatten(self.results)
        results.update(new_results)

        results = self.unflatten(results)
        self.results.update(results)

    @staticmethod
    def load_outputs(path):
        return np.load(path, allow_pickle=True)


def choose(a, b):
    if a is None:
        a = b
    else:
        b = a
    return a
