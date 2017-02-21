from __future__ import division
import numpy as np
from calib.models.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.base import clone
from functions import cross_entropy
from functions import brier_score
import timeit


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


def calibrate_timed(classifier, x_cali, y_cali, method=None, score_type=None):
    ccv = CalibratedClassifierCV(base_estimator=classifier, method=method,
                                 cv='prefit', score_type=score_type)
    tic = timeit.default_timer()
    ccv.fit(x_cali, y_cali)
    toc = timeit.default_timer()
    return ccv, toc-tic


# def calibrate_old(x_train, y_train, x_test, y_test, method=None, cv=3):
#     cccv = CalibratedClassifierCV(base_estimator=GaussianNB(), method=method,
#                                   cv=cv)
#     cccv.fit(x_train, y_train)
#     c_probas = cccv.predict_proba(x_test)[:, 1]
#     predictions = cccv.predict(x_test)
#     acc = np.mean(predictions == y_test)
#     loss = cross_entropy(c_probas, y_test)
#     return acc, loss, c_probas


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
                if verbose == True:
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


def cv_calibration_timed(base_classifier, methods, x_train, y_train, x_test,
                         y_test, cv=3, score_type=None):
    folds = StratifiedKFold(y_train, n_folds=cv, shuffle=True)
    mean_probas = {method: np.zeros(np.alen(y_test)) for method in methods}
    classifiers = {method: [] for method in methods}
    mean_times = {method: 0.0 for method in methods}
    for i, (train, cali) in enumerate(folds):
        if i < cv:
            x_t = x_train[train]
            y_t = y_train[train]
            x_c = x_train[cali]
            y_c = y_train[cali]
            classifier = clone(base_classifier)
            tic = timeit.default_timer()
            classifier.fit(x_t, y_t)
            toc = timeit.default_timer()
            mean_times[None] += (toc - tic) / cv
            for method in methods:
                ccv, time = calibrate_timed(classifier, x_c, y_c, method=method,
                                            score_type=score_type)
                if method is not None:
                    mean_times[method] += time / cv
                mean_probas[method] += ccv.predict_proba(x_test)[:, 1] / cv
                classifiers[method].append(ccv)
    losses = {method: cross_entropy(mean_probas[method], y_test) for method
              in methods}
    accs = {method: np.mean((mean_probas[method] >= 0.5) == y_test) for method
            in methods}
    briers = {method: brier_score(mean_probas[method], y_test) for method
              in methods}
    return accs, losses, briers, mean_probas, classifiers, mean_times


def map_calibration(base_classifier, methods, x_train, y_train, score_type=None):
    folds = StratifiedKFold(y_train, n_folds=2, shuffle=True).test_folds
    classifiers = {method: [] for method in methods}
    train = folds == 0
    cali = folds == 1
    x_t = x_train[train]
    y_t = y_train[train]
    x_c = x_train[cali]
    y_c = y_train[cali]
    probas = {method: np.zeros(np.alen(y_c)) for method in methods}
    classifier = clone(base_classifier)
    classifier.fit(x_t, y_t)
    for method in methods:
        ccv = calibrate(classifier, x_c, y_c, method=method,
                        score_type=score_type)
        probas[method] += ccv.predict_proba(x_c)[:, 1]
        classifiers[method].append(ccv)
    return probas, classifiers, y_c
