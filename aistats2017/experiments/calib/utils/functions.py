import numpy as np
from scipy.stats import rankdata


def cross_entropy(y_hat, y):
    y_hat = np.clip(y_hat, 1e-16, 1 - 1e-16)
    cs = -y * np.log(y_hat) - (1.0 - y) * np.log(1.0 - y_hat)
    return np.mean(cs)


def brier_score(y_hat, y):
    y_hat = np.clip(y_hat, 0.0, 1.0)
    return np.power(y_hat - y, 2.0).mean()


def get_sets(x, y, test_fold_id, test_folds):
    x_test = x[test_folds == test_fold_id, :]
    y_test = y[test_folds == test_fold_id]

    train_indices = test_folds != test_fold_id
    x_train = x[train_indices, :]
    y_train = y[train_indices]

    return [x_train, y_train, x_test, y_test]


# TODO: MPN need to join *to_latex functions
def table_to_latex(datasets, methods, table, max_is_better=True, caption=''):
    means = table.as_matrix()[:, :len(methods)]
    avg_ranks = np.zeros(len(methods))
    if max_is_better:
        means *= 100.0
    stds = table.as_matrix()[:, len(methods):]
    if max_is_better:
        stds *= 100.0
    print '\\begin{table}[!t]'
    print '\\centering'
    str_columns = 'l'
    str_header = 'dataset'
    methods.sort()
    for method in methods:
        str_columns += 'c'
        if method is None:
            str_header += ' & uncalibrated'
        else:
            str_header += ' & ' + method
    str_header += '\\\\'
    print '\\begin{tabular}{'+str_columns+'}'
    print '\\toprule'
    print str_header
    print '\\midrule'
    for i, name in enumerate(datasets):
        nam = name[:7] if len(name) > 7 else name
        str_row_means = nam
        str_row_stds = ''
        v = means[i]
        v_std = stds[i]
        indices = rankdata(v)
        if max_is_better:
            indices = len(methods) + 1 - indices
        for j in np.arange(len(v)):
            idx = indices[j]
            avg_ranks[j] += idx / len(datasets)
            if idx == 1:
                str_row_means += ' & $\\textbf{'+'{0:.3f}'.format(v[j])+'}_1$'
                str_row_stds += ' & (\\textbf{'+'{0:.3f}'.format(v_std[j])+'})'
            else:
                idx_s = '{}'.format(idx)
                if '.0' in idx_s:
                    idx_s = '{}'.format(int(idx))
                str_row_means += ' & ${0:.3f}'.format(v[j])+'_{' + idx_s + '}$'
                str_row_stds += ' & ({0:.3f}'.format(v_std[j])+")"
        print str_row_means + '\\\\'
        print str_row_stds + '\\\\'
    print '\\midrule'
    str_avg = 'rank'
    for i in np.arange(len(methods)):
        str_avg += ' & {0:.2f}'.format(avg_ranks[i])
    print str_avg + '\\\\'
    print "\\bottomrule"
    print "\\end{tabular}"
    print "\\caption{\\small{"+caption+"}}"
    print "\\label{table:table}"
    print "\\end{table}"


# TODO: MPN need to join *to_latex functions
def to_latex(datasets, table, max_is_better=True, scale=1, precision=3,
             table_size="\\normalsize", caption=''):
    column_names = table.columns.levels[2]
    n_columns = len(column_names)
    row_names = table.index
    n_rows = len(row_names)

    means = table.as_matrix()[:, :n_columns]*scale
    avg_ranks = np.zeros(n_columns)
    stds = table.as_matrix()[:, n_columns:]*scale
    str_table = ('\\begin{table}[!t]\n' +
                 table_size + '\n' +
                 '\\centering\n')
    str_columns = 'l'
    str_header = ''

    for c_name in column_names:
        str_columns += 'c'
        str_header += ' & ' + c_name
    str_header += '\\\\\n'

    str_table += ('\\begin{tabular}{'+str_columns+'}\n' +
                  '\\toprule\n' +
                  str_header +
                  '\\midrule\n')
    for i, name in enumerate(row_names):
        name = name[:10] if len(name) > 10 else name
        name = name.replace('_', r'\_')
        str_row_means = name
        str_row_stds = ''
        v = means[i]
        v_std = stds[i]
        indices = rankdata(v)
        if max_is_better:
            indices = n_columns + 1 - indices
        for j in np.arange(len(v)):
            idx = indices[j]
            avg_ranks[j] += idx / n_rows
            if idx == 1:
                #str_row_means += (' & $\\textbf{{{0:.{1}f}}}_1$'.format(v[j],
                #                  precision))
                #str_row_stds += (' & (\\textbf{{\\tiny{{{0:.{1}f}}}}})'.format(v_std[j],
                #                 precision))
                str_row_means += (' & $\\mathbf{{{0:.{2}f}\\pm{1:.{2}f}_{{{3}}}}}$'.format(
                                    v[j], v_std[j], precision, 1))
            else:
                idx_s = '{}'.format(idx)
                if '.0' in idx_s:
                    idx_s = '{}'.format(int(idx))
                str_row_means += (' & ${0:.{2}f}\\pm{1:.{2}f}_{{{3}}}$'.format(
                                    v[j], v_std[j], precision, idx_s))
                # two lines format
                #str_row_stds += (' & \\tiny{{$({0:.{1}f})$}}'.format(v_std[j],
                #                 precision))
        str_table += str_row_means + '\\\\\n'
        #str_table += str_row_stds + '\\\\\n'
    str_table += '\\midrule\n'
    str_avg = 'average rank'
    for i in np.arange(n_columns):
        str_avg += ' & {0:.2f}'.format(avg_ranks[i])

    str_table += (str_avg + '\\\\\n' +
                  '\\bottomrule\n' +
                  '\\end{tabular}\n' +
                  '\\caption{\\small{'+caption+'}}\n' +
                  '\\label{table:table}\n' +
                  '\\end{table}\n')
    return str_table


