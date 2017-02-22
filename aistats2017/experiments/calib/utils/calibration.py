from __future__ import division
import numpy as np
from calib.models.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.base import clone
from functions import cross_entropy
from functions import brier_score


def get_calibrated_scores(classifiers, methods, scores):
    probas = []
    for method in methods:
        p = np.zeros(len(scores))
        c_method = classifiers[method]
        for classifier in c_method:
            p += classifier.calibrate_scores(scores)[:, 1]
        probas.append(p / np.alen(c_method))
    return probas


def calibrate(classifier, x_cali, y_cali, method=None, score_type=None):
    ccv = CalibratedClassifierCV(base_estimator=classifier, method=method,
                                 cv='prefit', score_type=score_type)
    ccv.fit(x_cali, y_cali)
    return ccv


def cv_calibration(base_classifier, methods, x_train, y_train, x_test, y_test,
                   cv=3, score_type=None, verbose=False):
    folds = StratifiedKFold(y_train, n_folds=cv, shuffle=True)
    mean_probas = {method: np.zeros(np.alen(y_test)) for method in methods}
    classifiers = {method: [] for method in methods}
    for i, (train, cali) in enumerate(folds):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            classifier.fit(x_t, y_t)
            for method in methods:
                if verbose:
                    print("Calibrating with " + 'none' if method is None else
                          method)
                ccv = calibrate(classifier, x_c, y_c, method=method,
                                score_type=score_type)
                mean_probas[method] += ccv.predict_proba(x_test)[:, 1] / cv
                classifiers[method].append(ccv)
    losses = {method: cross_entropy(mean_probas[method], y_test) for method
              in methods}
    accs = {method: np.mean((mean_probas[method] >= 0.5) == y_test) for method
            in methods}
    briers = {method: brier_score(mean_probas[method], y_test) for method
              in methods}
    return accs, losses, briers, mean_probas, classifiers
