"""Summarizes training results"""

import fcntl
import pandas as pd
from pathlib import Path


VAL = "val"


class Summary:
    def __init__(self, histories):
        """Summarize training run.

        Args:
            histories ([type]): model.fit history.history dict.
        """

        if type(histories) is list:
            self.histories = histories
        else:
            self.histories = [histories]

    def summarize_eval(self):
        folds = {"fold": {}}

        for i, history in enumerate(self.histories):
            folds["fold"][i] = history

        average = to_dataframe(folds["fold"]).mean(axis=1)

        folds["fold"]["avg"] = {}
        folds["fold"]["avg"]["last"] = {k: v for k, v in average.items()}

        return folds

    def summarize_best_history(self, valid_freq, name="val_loss"):
        assert name is None or VAL in name, f"param name must contain `{VAL}`"
        assert valid_freq > 0, f"valid_freq={valid_freq} but must be > 0"

        folds = {"fold": {}}

        for i, history in enumerate(self.histories):
            df = to_dataframe(history)
            folds["fold"][i] = self.best(df, valid_freq, name)

        average = to_dataframe(folds["fold"]).mean(axis=1)

        folds["fold"]["avg"] = {}
        folds["fold"]["avg"]["best"] = {k: v for k, v in average.items()}

        return folds

    def summarize_last_history(self, valid_freq):
        assert valid_freq > 0, f"valid_freq={valid_freq} but must be > 0"

        folds = {"fold": {}}

        for i, history in enumerate(self.histories):
            df = to_dataframe(history)
            folds["fold"][i] = self.last(df)

        average = to_dataframe(folds["fold"]).mean(axis=1)

        folds["fold"]["avg"] = {}
        folds["fold"]["avg"]["last"] = {k: v for k, v in average.items()}

        return folds

    def per_fold(self, history, valid_freq, name="val_loss"):
        df = to_dataframe(history)
        last = self.last(df)
        best = self.best(df, valid_freq, name)

        return {"best": best, "last": last}

    def best(self, df, valid_freq, name):
        if "loss" in name:
            best_ind = get_min_index(df, name=name)
        else:
            best_ind = get_max_index(df, name=name)

        adjusted_ind = valid_freq * (best_ind + 1) - 1

        best = {}
        best["epoch"] = adjusted_ind
        best.update(
            {k: (df[k][best_ind] if VAL in k else df[k][adjusted_ind]) for k in df}
        )

        return best

    def last(self, df):
        last_ind = {k: df[k].last_valid_index() for k in df}

        last = {}
        last["epoch"] = last_ind["loss"]
        last.update({k: df[k][last_ind[k]] for k in df})

        return last


def get_max_index(df, name=None):
    if name is None or name not in df:
        return df.idxmax(axis=0, skipna=True)
    return df.idxmax(axis=0, skipna=True)[name]


def get_min_index(df, name=None):
    if name is None or name not in df:
        return df.idxmin(axis=0, skipna=True)
    return df.idxmin(axis=0, skipna=True)[name]


def to_dataframe(results):
    return pd.DataFrame.from_dict(results, orient="index").transpose()


def thread_safe_csv_writes(df, path, mode, header):
    with open(path, mode) as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        df.to_csv(f, mode=mode, header=header, index_label="run_id", index=False)
        fcntl.flock(f, fcntl.LOCK_UN)


def save_results(hyperparams):
    df = pd.DataFrame([hyperparams])
    path = hyperparams["env/path/param_csv"].replace("hyper.csv", "results.csv")
    csv_path = Path(path)

    # --- Save csv
    if not csv_path.is_file():
        thread_safe_csv_writes(df, path, mode="w", header=True)
    else:
        thread_safe_csv_writes(df, path, mode="a", header=False)

    print(f"- csv saved to {path}")
