import numpy as np
import matplotlib.pyplot as plt

def reliability_diagram(prob, Y, marker='--', label='', alpha=1, linewidth=1,
                        ax_reliability=None, clip=True):
    '''
        alpha= Laplace correction, default add-one smoothing
    '''
    bins = np.linspace(0,1+1e-16,11)
    prob = np.clip(prob, 0, 1)
    hist_tot = np.histogram(prob, bins=bins)
    hist_pos = np.histogram(prob[Y == 1], bins=bins)
    # Compute the centroids of every bin
    centroids = [np.mean(np.append(
                 prob[np.where(np.logical_and(prob >= bins[i],
                                              prob < bins[i+1]))],
                 bins[i]+0.05)) for i in range(len(hist_tot[1])-1)]

    proportion = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+alpha*2)
    if ax_reliability is None:
        ax_reliability = plt.subplot(111)

    ax_reliability.plot(centroids, proportion, marker, linewidth=linewidth,
                        label=label)


def plot_reliability_diagram(scores_set, labels, legend_set,
                             original_first=False, alpha=1, **kwargs):
    fig_reliability = plt.figure('reliability_diagram')
    fig_reliability.clf()
    ax_reliability = plt.subplot(111)
    ax = ax_reliability
    # ax.set_title('Reliability diagram')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    n_lines = len(legend_set)
    if original_first:
        bins = np.linspace(0, 1, 11)
        hist_tot = np.histogram(scores_set[0], bins=bins)
        hist_pos = np.histogram(scores_set[0][labels == 1], bins=bins)
        edges = np.insert(bins, np.arange(len(bins)), bins)
        empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
        empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                                empirical_p)
        ax.plot(edges[1:-1], empirical_p, label='empirical')

    skip = original_first
    for (scores, legend) in zip(scores_set, legend_set):
        if skip and original_first:
            skip = False
        else:
            reliability_diagram(scores, labels, marker='x-',
                    label=legend, linewidth=n_lines, alpha=alpha, **kwargs)
            n_lines -= 1
    if original_first:
        ax.plot(scores_set[0], labels, 'kx', label=legend_set[0],
                markersize=9, markeredgewidth=1)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig_reliability


def plot_reliability_map(scores_set, prob, legend_set,
                         original_first=False, alpha=1, **kwargs):
    fig_reliability_map = plt.figure('reliability_map')
    fig_reliability_map.clf()
    ax_reliability_map = plt.subplot(111)
    ax = ax_reliability_map
    # ax.set_title('Reliability map')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    n_lines = len(legend_set)
    if original_first:
        bins = np.linspace(0, 1, 11)
        hist_tot = np.histogram(prob[0], bins=bins)
        hist_pos = np.histogram(prob[0][prob[1] == 1], bins=bins)
        edges = np.insert(bins, np.arange(len(bins)), bins)
        empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
        empirical_p = np.insert(empirical_p, np.arange(len(empirical_p)),
                                empirical_p)
        ax.plot(edges[1:-1], empirical_p, label='empirical')

    skip = original_first
    for (scores, legend) in zip(scores_set, legend_set):
        if skip and original_first:
            skip = False
        else:
            if legend == 'uncalib':
                ax.plot([np.nan], [np.nan], '-', linewidth=n_lines,
                        **kwargs)
            else:
                ax.plot(prob[2], scores, '-', label=legend, linewidth=n_lines,
                        **kwargs)
            n_lines -= 1
    if original_first:
        ax.plot(prob[0], prob[1], 'kx',
                label=legend_set[0], markersize=9, markeredgewidth=1)
    ax.legend(loc='upper left')
    ax.grid(True)
    return fig_reliability_map


def plot_niculescu_mizil_map(scores_set, prob, legend_set, alpha=1, **kwargs):
    from matplotlib import rc
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    fig_reliability_map = plt.figure('reliability_map')
    fig_reliability_map.clf()
    ax_reliability_map = plt.subplot(111)
    ax = ax_reliability_map
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_xlabel((r'$s$'), fontsize=16)
    ax.set_ylabel((r'$\hat{p}$'), fontsize=16)
    n_lines = len(legend_set)
    bins = np.linspace(0, 1, 11)
    hist_tot = np.histogram(prob[0], bins=bins)
    hist_pos = np.histogram(prob[0][prob[1] == 1], bins=bins)
    centers = (bins[:-1] + bins[1:])/2.0
    empirical_p = np.true_divide(hist_pos[0]+alpha, hist_tot[0]+2*alpha)
    ax.plot(centers, empirical_p, 'ko', label='empirical')

    for (scores, legend) in zip(scores_set, legend_set):
        if legend != 'uncalib':
            ax.plot(prob[2], scores, '-', label=legend, linewidth=n_lines,
                    **kwargs)
        n_lines -= 1
    ax.legend(loc='upper left')
    return fig_reliability_map


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    np.random.seed(42)
    # Random scores
    n = np.random.normal(loc=-4, scale=2, size=100)
    p = np.random.normal(loc=4, scale=2, size=100)
    s = np.append(n, p)
    plt.hist(s)
    plt.show()
    s.sort()
    s1 = s.reshape(-1, 1)

    # Obtaining probabilities from the scores
    s1 = sigmoid(s1)
    # Obtaining the two features for beta-calibration with 3 parameters
    s1 = np.log(np.hstack((s1, 1.0 - s1)))
    # s1[:, 1] *= -1

    # Generating random labels
    y = np.append(np.random.binomial(1, 0.1, 40), np.random.binomial(1, 0.3,
                                                                     40))
    y = np.append(y, np.random.binomial(1, 0.4, 40))
    y = np.append(y, np.random.binomial(1, 0.4, 40))
    y = np.append(y, np.ones(40))

    # Fitting Logistic Regression without regularization
    lr = LogisticRegression(C=99999999999)
    lr.fit(s1, y)

    linspace = np.linspace(-10, 10, 100)
    l = sigmoid(linspace).reshape(-1, 1)
    l1 = np.log(np.hstack((l, 1.0 - l)))
    # l1[:, 1] *= -1

    probas = lr.predict_proba(l1)[:, 1]
    s_exp = sigmoid(s)
    fig_map = plot_niculescu_mizil_map([probas], [s_exp, y, l],
                                       ['beta'], alpha=1)

    plt.show()

