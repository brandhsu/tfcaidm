"""Model evaluation interface"""

import numpy as np
from scipy import stats
from scipy.spatial import distance
from sklearn import metrics


class Base:
    def __init__(self, model):
        self.model = model

    def forward(self, x):
        k = self.model.output_names
        v = self.model.predict(x)
        outputs = {k[i]: v[k[i]] for i in range(len(k))}

        return outputs

    def sort(self, x, best="high"):
        if best == "high":
            i = np.argsort(-x)
        elif best == "low":
            i = np.argsort(x)
        else:
            raise ValueError("arg `best` must be either 'high' or 'low'")

        return i

    @staticmethod
    def flatten(func):
        def apply(self, y_true, y_pred, *args, **kwargs):
            return func(self, y_true.flatten(), y_pred.flatten(), *args, **kwargs)

        return apply


class Classifier(Base):
    def __init__(self, model):
        super(Classifier, self).__init__(model)

    @Base.flatten
    def balanced_accuracy_score(
        self, y_true, y_pred, *, sample_weight=None, adjusted=False
    ):
        """Same as sklearn.metrics.balanced_accuracy_score"""

        return metrics.balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weight, adjusted=adjusted
        )

    @Base.flatten
    def classification_report(
        self,
        y_true,
        y_pred,
        *,
        labels=None,
        target_names=None,
        sample_weight=None,
        digits=2,
        output_dict=False,
        zero_division="warn",
    ):
        """Same as sklearn.metrics.classification_report"""

        return metrics.classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            sample_weight=sample_weight,
            digits=digits,
            output_dict=output_dict,
            zero_division=zero_division,
        )

    @Base.flatten
    def confusion_matrix(
        self, y_true, y_pred, *, labels=None, sample_weight=None, normalize=None
    ):
        """Same as sklearn.metrics.confusion_matrix"""

        return metrics.confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            sample_weight=sample_weight,
            normalize=normalize,
        )

    @Base.flatten
    def jaccard_score(
        self,
        y_true,
        y_pred,
        *,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    ):
        """Same as sklearn.metrics.jaccard_score"""

        return metrics.jaccard_score(
            y_true,
            y_pred,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    @Base.flatten
    def f1_score(
        self,
        y_true,
        y_pred,
        *,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    ):
        """Same as sklearn.metrics.f1_score"""

        return metrics.f1_score(
            y_true,
            y_pred,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

    @Base.flatten
    def matthews_corrcoef(self, y_true, y_pred, *, sample_weight=None):
        """Same as sklearn.metrics.matthews_corrcoef"""

        return metrics.matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)


class Regressor(Base):
    def __init__(self, model):
        super(Regressor, self).__init__(model)

    def cdist(self, XA, XB, metric="euclidean", *, out=None, **kwargs):

        """Same as scipy.spatial.distance.cdist"""

        return distance.cdist(XA, XB, metric=metric, out=out, **kwargs)

    @Base.flatten
    def mean_absolute_error(
        self, y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
    ):
        """Same as sklearn.metrics.mean_absolute_error"""

        return metrics.mean_absolute_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )

    @Base.flatten
    def mean_squared_error(
        self,
        y_true,
        y_pred,
        *,
        sample_weight=None,
        multioutput="uniform_average",
        squared=True,
    ):
        """Same as sklearn.metrics.mean_squared_error"""

        return metrics.mean_squared_error(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
            squared=squared,
        )

    @Base.flatten
    def pearsonr(self, y_true, y_pred):
        """Same as scipy.stats.pearsonr"""

        return stats.pearsonr(y_true, y_pred)

    @Base.flatten
    def r2_score(
        self, y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
    ):
        """Same as sklearn.metrics.r2_score"""

        return metrics.r2(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
        )
