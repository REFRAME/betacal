# Usage:
# Parallelized in multiple threads:
#   python -m scoop -n 4 main_nb.py # where -n is the number of workers (threads)
# Not parallelized (easier to debug):
#   python main_nb.py

from __future__ import division
import os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
import calib.models.adaboost as our
import sklearn.ensemble as their
from scipy.stats import friedmanchisquare
# Parallelization
import itertools
from scoop import futures

# Our classes and modules
from calib.utils.calibration import get_calibrated_scores
from calib.utils.calibration import map_calibration
from calib.utils.dataframe import MyDataFrame
from calib.utils.functions import cross_entropy
from calib.utils.functions import get_sets
from calib.utils.functions import table_to_latex
from calib.utils.plots import plot_reliability_diagram
from calib.utils.plots import plot_reliability_map
from calib.utils.plots import plot_niculescu_mizil_map

# Our datasets module
from data_wrappers.datasets import Data
from data_wrappers.datasets import dataset_names_binary
from data_wrappers.datasets import datasets_li2014
from data_wrappers.datasets import datasets_hempstalk2008
from data_wrappers.datasets import datasets_others
from calib.models.adaboost import AdaBoostClassifier

methods = [None, 'sigmoid', 'isotonic', 'beta']
seed_num = 42
mc_iterations = 10
n_folds = 5
results_path = 'maps'


def compute_all(args):
    (name, dataset, n_folds, mc) = args
    np.random.seed(mc)
    skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                          shuffle=True)
    test_folds = skf.test_folds
    class_counts = np.bincount(dataset.target)
    classifiers = None
    c_probas = None
    y_c = None
    mc_save = 0
    test_fold_save = 0
    if np.alen(class_counts) > 2:
        majority = np.argmax(class_counts)
        t = np.zeros_like(dataset.target)
        t[dataset.target == majority] = 1
    else:
        t = dataset.target
    for test_fold in np.arange(n_folds):
        x_train, y_train, x_test, y_test = get_sets(dataset.data,
                                                    t,
                                                    test_fold,
                                                    test_folds)
        probas, cl, yc = map_calibration(our.AdaBoostClassifier(n_estimators=200), methods, x_train, y_train)
        if mc == mc_save and test_fold == test_fold_save:
            classifiers = cl
            c_probas = probas
            y_c = yc

    if mc == mc_save:
        test_fold = test_fold_save
        methods_text = ['uncalib' if m is None else m for m in methods]

        probas = [c_probas[method] for method in methods]

        linspace = np.linspace(0, 1, 100)
        scores = probas[0]
        idx = scores.argsort()
        scores = scores[idx]
        y_c_2 = y_c[idx]

        probas = get_calibrated_scores(classifiers, methods, linspace)

        fig_map = plot_niculescu_mizil_map(probas, [scores, y_c_2, linspace], methods_text, alpha=0)

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        fig_map.savefig(os.path.join(results_path,
                        '{}_mc{}_fold{}_map.pdf'.format(name, mc, test_fold)))
        exit()


if __name__ == '__main__':
    dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
                             datasets_others))
    dataset_names = ['vowel']
    dataset_names.sort()

    data = Data(dataset_names=dataset_names)
    # for name, dataset in data.datasets.iteritems():
    #     if name in ['letter', 'shuttle']:
    #         dataset.reduce_number_instances(0.1)

    for name, dataset in data.datasets.iteritems():
        print(dataset)

        mcs = np.arange(mc_iterations)
        # All the arguments as a list of lists
        args = [[name], [dataset], [n_folds], mcs]
        args = list(itertools.product(*args))

        # if called with -m scoop
        if '__loader__' in globals():
            futures.map(compute_all, args)
        else:
            map(compute_all, args)
