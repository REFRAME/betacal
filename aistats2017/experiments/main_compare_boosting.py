# Usage:
# Parallelized in multiple threads:
#   python -m scoop -n 4 main_boosting.py # where -n is the number of workers (
# threads)
# Not parallelized (easier to debug):
#   python main_boosting.py

from __future__ import division
import os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import friedmanchisquare

from calib.models import adaboost
from sklearn import ensemble

# Parallelization
import itertools
from scoop import futures

# Our classes and modules
from calib.utils.dataframe import MyDataFrame
from calib.utils.functions import get_sets
from calib.utils.functions import table_to_latex

# Our datasets module
from data_wrappers.datasets import Data
from data_wrappers.datasets import datasets_li2014
from data_wrappers.datasets import datasets_hempstalk2008
from data_wrappers.datasets import datasets_others

from calib.utils.functions import cross_entropy
from calib.utils.functions import brier_score

import matplotlib.pyplot as plt

methods = ['Our Ada', 'Their Ada']
seed_num = 42
mc_iterations = 10
n_folds = 5

n_estimators = 1000

columns = ['dataset', 'method', 'mc', 'test_fold', 'acc', 'loss', 'brier',
           'c_probas']


def compute_all(args):
    (name, dataset, n_folds, mc) = args
    np.random.seed(mc)
    skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                          shuffle=True)
    df = MyDataFrame(columns=columns)
    test_folds = skf.test_folds
    class_counts = np.bincount(dataset.target)
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

        our_ada = adaboost.AdaBoostClassifier(n_estimators=n_estimators)
        our_ada.fit(x_train, y_train)
        our_probas = our_ada.predict_proba(x_test)[:, 1]
        our_acc = np.mean((our_probas >= 0.5) == y_test)
        our_loss = cross_entropy(our_probas, y_test)
        our_brier = brier_score(our_probas, y_test)

        df = df.append_rows([[name, methods[0], mc, test_fold, our_acc,
                              our_loss, our_brier, our_probas]])

        their_ada = ensemble.AdaBoostClassifier(n_estimators=n_estimators)
        their_ada.fit(x_train, y_train)
        their_probas = their_ada.predict_proba(x_test)[:, 1]
        their_acc = np.mean((their_probas >= 0.5) == y_test)
        their_loss = cross_entropy(their_probas, y_test)
        their_brier = brier_score(their_probas, y_test)

        df = df.append_rows([[name, methods[1], mc, test_fold, their_acc,
                              their_loss, their_brier, their_probas]])
        plt.hist(our_probas, 10, color='blue')
        plt.hist(their_probas, 10, color='green')
        plt.show()
    return df


if __name__ == '__main__':
    dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
                             datasets_others))
    dataset_names = ['hepatitis']
    dataset_names.sort()
    df_all = MyDataFrame(columns=columns)

    data = Data(dataset_names=dataset_names)
    # for name, dataset in data.datasets.iteritems():
    #     if name in ['letter', 'shuttle']:
    #         dataset.reduce_number_instances(0.1)

    for name, dataset in data.datasets.iteritems():
        df = MyDataFrame(columns=columns)
        print(dataset)

        mcs = np.arange(mc_iterations)
        # All the arguments as a list of lists
        args = [[name], [dataset], [n_folds], mcs]
        args = list(itertools.product(*args))

        # if called with -m scoop
        if '__loader__' in globals():
            dfs = futures.map(compute_all, args)
        else:
            dfs = map(compute_all, args)

        df = df.concat(dfs)

        table = df[df.dataset == name].pivot_table(values=['acc', 'loss',
            'c_probas'], index=['method'], aggfunc=[np.mean, np.std])

        print(table)
        print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        df_all = df_all.append(df)

    table = df_all.pivot_table(index=['dataset'], columns=['method'],
                               values=['acc'], aggfunc=[np.mean, np.std])
    table_to_latex(dataset_names, methods, table, max_is_better=True)
    accs = table.as_matrix()[:, :4]
    print friedmanchisquare(*[accs[:, x] for x in np.arange(accs.shape[1])])

    table = df_all.pivot_table(index=['dataset'], columns=['method'],
                               values=['loss'], aggfunc=[np.mean, np.std])
    table_to_latex(dataset_names, methods, table, max_is_better=False)
    losses = table.as_matrix()[:, :4]
    print friedmanchisquare(*[losses[:, x] for x in np.arange(losses.shape[1])])

    table = df_all.pivot_table(index=['dataset'], columns=['method'],
                               values=['brier'], aggfunc=[np.mean, np.std])
    table_to_latex(dataset_names, methods, table, max_is_better=False)
    briers = table.as_matrix()[:, :4]
    print friedmanchisquare(*[briers[:, x] for x in np.arange(briers.shape[1])])
