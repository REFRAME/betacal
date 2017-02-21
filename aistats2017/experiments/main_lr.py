# Usage:
# Parallelized in multiple threads:
#   python -m scoop -n 4 main_nb.py # where -n is the number of workers (threads)
# Not parallelized (easier to debug):
#   python main_nb.py

from __future__ import division
import os
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import friedmanchisquare
# Parallelization
import itertools
from scoop import futures

# Our classes and modules
from calib.utils.calibration import get_calibrated_scores
from calib.utils.calibration import calibrate
from calib.utils.calibration import cv_calibration
from calib.utils.calibration import cv_calibration_timed
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

methods = [None, 'sigmoid', 'isotonic', 'beta', 'beta2', 'beta05']
#methods = [None, 'sigmoid', 'beta', 'beta2']
seed_num = 42
mc_iterations = 10
n_folds = 5
results_path = 'results'

columns = ['dataset', 'method', 'mc', 'test_fold', 'acc', 'loss', 'brier',
           'time', 'c_probas']


def compute_all(args):
    (name, dataset, n_folds, mc) = args
    np.random.seed(mc)
    skf = StratifiedKFold(dataset.target, n_folds=n_folds,
                          shuffle=True)
    df = MyDataFrame(columns=columns)
    test_folds = skf.test_folds
    class_counts = np.bincount(dataset.target)
    classifiers = None
    mc_save = 6
    test_fold_save = 1
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
        # accs, losses, briers, mean_probas, cl = cv_calibration(LogisticRegression(),
        #                                              methods, x_train, y_train,
        #                                              x_test, y_test, cv=3,
        #                                              score_type='predict_proba')
        accs, losses, briers, mean_probas, cl, times = cv_calibration_timed(
            LogisticRegression(),
                                                     methods, x_train, y_train,
                                                     x_test, y_test, cv=3,
                                                     score_type='predict_proba')
        if mc == mc_save and test_fold == test_fold_save:
            classifiers = cl
        for method in methods:
            m_text = 'None' if method is None else method
            df = df.append_rows([[name, m_text, mc, test_fold,
                                  accs[method], losses[method], briers[method],
                                  times[method], mean_probas[method]]])
        # for method in methods:
        #     acc, loss, c_probas = calibrate(
        #             x_train, y_train, x_test, y_test, method=method)
        #     method = 'None' if method is None else method
        #     df = df.append_rows([[name, method, mc, test_fold,
        #                           acc, loss, c_probas]])

    # if mc == mc_save:
    #     test_fold = test_fold_save
    #     x_train, y_train, x_test, y_test = get_sets(dataset.data,
    #                                                 t,
    #                                                 test_fold,
    #                                                 test_folds)
    #     methods_text = ['uncalib' if m is None else m for m in methods]
    #     probas = df[df.test_fold == test_fold].c_probas
    #     probas = [p for p in probas]
    #     fig = plot_reliability_diagram(probas, y_test, methods_text,
    #                                    original_first=True)
    #     if not os.path.exists(results_path):
    #         os.makedirs(results_path)
    #     fig.savefig(os.path.join(results_path,
    #                 '{}_mc{}_fold{}.pdf'.format(name, mc, test_fold)))
    #
    #     linspace = np.linspace(0, 1, 100)
    #     scores = [s for s in df[np.logical_and(df.test_fold == test_fold,
    #                                df.method == 'None')].c_probas][0]
    #     idx = scores.argsort()
    #     scores = scores[idx]
    #     y_test_2 = y_test[idx]
    #
    #     probas = get_calibrated_scores(classifiers, methods, linspace)
    #     # scores = [scores for p in len(methods_text)]
    #
    #     fig_map = plot_reliability_map(probas, [scores, y_test_2, linspace],
    #                                    methods_text,
    #                                    original_first=False, alpha=1)
    #
    #     if not os.path.exists(results_path):
    #         os.makedirs(results_path)
    #     fig_map.savefig(os.path.join(results_path,
    #                     '{}_mc{}_fold{}_map.pdf'.format(name, mc, test_fold)))
    return df


if __name__ == '__main__':
    dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
                             datasets_others))
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

        table = df[df.dataset == name].pivot_table(values=['acc', 'loss', 'brier',
            'time', 'c_probas'], index=['method'], aggfunc=[np.mean, np.std])

        print(table)
        print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        df_all = df_all.append(df)
    table = df_all.pivot_table(values=['acc', 'loss'], index=['dataset', 'method'],
                           aggfunc=[np.mean, np.std])
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df_all.to_csv(os.path.join(results_path, 'main_results_data_frame.csv'))

    table.to_csv(os.path.join(results_path, 'main_results.csv'))
    table.to_latex(os.path.join(results_path, 'main_results.tex'))

    remove_list = [[], ['isotonic'], ['beta2'], ['beta05'], ['beta', 'beta05'], ['beta2', 'beta05'],
                   [None, 'None', 'isotonic', 'sigmoid']]
    for rem in remove_list:
        df_rem = df_all[np.logical_not(np.in1d(df_all.method, rem))]
        methods_rem = [method for method in methods if method not in rem]
        print methods_rem
        print("-#-#-#-#-#-#-#-#-#-#-#-#-ACC-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                   values=['acc'], aggfunc=[np.mean, np.std])
        table_to_latex(dataset_names, methods_rem, table, max_is_better=True)
        accs = table.as_matrix()[:, :len(methods_rem)]
        print friedmanchisquare(*[accs[:, x] for x in np.arange(accs.shape[1])])
        table.to_csv(os.path.join(results_path, 'main_acc' + str(methods_rem) + '.csv'))

        print("-#-#-#-#-#-#-#-#-#-#-#-LOSS-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                   values=['loss'], aggfunc=[np.mean, np.std])
        table_to_latex(dataset_names, methods_rem, table, max_is_better=False)
        losses = table.as_matrix()[:, :len(methods_rem)]
        print friedmanchisquare(*[losses[:, x] for x in np.arange(losses.shape[1])])
        table.to_csv(os.path.join(results_path, 'main_loss' + str(methods_rem) + '.csv'))

        print("-#-#-#-#-#-#-#-#-#-#-#-BRIER-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                   values=['brier'], aggfunc=[np.mean, np.std])
        table_to_latex(dataset_names, methods_rem, table, max_is_better=False)
        briers = table.as_matrix()[:, :len(methods_rem)]
        print friedmanchisquare(*[briers[:, x] for x in np.arange(briers.shape[1])])
        table.to_csv(os.path.join(results_path, 'main_brier' + str(methods_rem) + '.csv'))

        print("-#-#-#-#-#-#-#-#-#-#-#-TIME-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
        table = df_rem.pivot_table(index=['dataset'], columns=['method'],
                                   values=['time'], aggfunc=[np.mean, np.std])
        table_to_latex(dataset_names, methods_rem, table, max_is_better=False)
        times = table.as_matrix()[:, :len(methods_rem)]
        print friedmanchisquare(*[times[:, x] for x in np.arange(times.shape[1])])
        table.to_csv(os.path.join(results_path, 'main_time' + str(methods_rem) +
                                  '.csv'))

        print("-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-")
