from __future__ import division

import warnings

import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm, dirichlet, gamma, t


from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import signature
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import check_cv

from beta_calibration import _BetaCalibration
from calibration import _SigmoidCalibration
from calibration import _DummyCalibration


class SSCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """Probability calibration with isotonic regression, sigmoid or beta.

    With this class, the base_estimator is fit on the train set of the
    cross-validation generator and the test set is used for calibration.
    The probabilities for each of the folds are then averaged
    for prediction. In case cv="prefit" is passed to __init__,
    it is assumed that base_estimator has been
    fitted already and all data is used for calibration. Note that
    data for fitting the classifier and for calibrating it must be disjoint.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv=prefit, the
        classifier must have been fit already on data.

    method : None, 'sigmoid', 'isotonic' or 'beta'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parameteric approach. It is not advised to use isotonic calibration
        with too few calibration samples ``(<<1000)`` since it tends to overfit.
        Use sigmoid (Platt's calibration) in this case.

    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If ``y`` is neither binary nor
        multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.

    Attributes
    ----------
    classes_ : array, shape (n_classes)
        The class labels.

    calibrated_classifiers_: list (len() equal to cv or 1 if cv == "prefit")
        The list of calibrated classifiers, one for each cross-validation fold,
        which has been fitted on all but the validation fold and calibrated
        on the validation fold.

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """
    def __init__(self, base_estimator=None, method=None, cv=3):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y, sample_weight=None):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo'],
                         force_all_finite=False)
        X, y = indexable(X, y)
        lb = LabelBinarizer().fit(y[y != -1])
        self.classes_ = lb.classes_

        # Check that each cross-validation fold can have at least one
        # example per class
        n_folds = self.cv if isinstance(self.cv, int) \
            else self.cv.n_folds if hasattr(self.cv, "n_folds") else None
        if n_folds and \
           np.any([np.sum(y == class_) < n_folds for class_ in self.classes_]):
            raise ValueError("Requesting %d-fold cross-validation but provided"
                             " less than %d examples for at least one class."
                             % (n_folds, n_folds))

        self.calibrated_classifiers_ = []
        if self.base_estimator is None:
            # we want all classifiers that don't expose a random_state
            # to be deterministic (and we don't want to expose this one).
            base_estimator = LinearSVC(random_state=0)
        else:
            base_estimator = self.base_estimator

        if self.cv == "prefit":
            calibrated_classifier = _CalibratedClassifier(
                base_estimator, method=self.method)
            if sample_weight is not None:
                calibrated_classifier.fit(X, y, sample_weight)
            else:
                calibrated_classifier.fit(X, y)
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            cv = check_cv(self.cv, X, y, classifier=True)
            fit_parameters = signature(base_estimator.fit).parameters
            estimator_name = type(base_estimator).__name__
            if (sample_weight is not None
                    and "sample_weight" not in fit_parameters):
                warnings.warn("%s does not support sample_weight. Sample"
                              " weights are only used for the calibration"
                              " itself." % estimator_name)
                base_estimator_sample_weight = None
            else:
                base_estimator_sample_weight = sample_weight
            for train, test in cv:
                this_estimator = clone(base_estimator)
                if base_estimator_sample_weight is not None:
                    this_estimator.fit(
                        X[train], y[train],
                        sample_weight=base_estimator_sample_weight[train])
                else:
                    this_estimator.fit(X[train], y[train])

                calibrated_classifier = _CalibratedClassifier(
                    this_estimator, method=self.method)
                x_test = X[test]
                y_test = y[test]
                if self.method != 'semi':
                    x_test = x_test[y_test != -1]
                    y_test = y_test[y_test != -1]
                if sample_weight is not None:
                    calibrated_classifier.fit(x_test, y_test,
                                              sample_weight[test])
                else:
                    calibrated_classifier.fit(x_test, y_test)
                self.calibrated_classifiers_.append(calibrated_classifier)

        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        check_is_fitted(self, ["classes_", "calibrated_classifiers_"])
        X = check_array(X, accept_sparse=['csc', 'csr', 'coo'],
                        force_all_finite=False)
        # Compute the arithmetic mean of the predictions of the calibrated
        # classifiers
        mean_proba = np.zeros((X.shape[0], len(self.classes_)))
        for calibrated_classifier in self.calibrated_classifiers_:
            proba = calibrated_classifier.predict_proba(X)
            mean_proba += proba

        mean_proba /= len(self.calibrated_classifiers_)

        return mean_proba

    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self, ["classes_", "calibrated_classifiers_"])
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _CalibratedClassifier(object):
    """Probability calibration with isotonic regression, sigmoid or beta.

    It assumes that base_estimator has already been fit, and trains the
    calibration on the input set of the fit function. Note that this class
    should not be used as an estimator directly. Use CalibratedClassifierCV
    with cv="prefit" instead.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. No default value since
        it has to be an already fitted estimator.

    method : 'sigmoid' | 'isotonic' | 'beta'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parameteric approach based on isotonic regression.

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """
    def __init__(self, base_estimator, method='sigmoid'):
        self.base_estimator = base_estimator
        self.method = method

    def _preproc(self, X):
        n_classes = len(self.classes_)
        if hasattr(self.base_estimator, "decision_function"):
            df = self.base_estimator.decision_function(X)
            if df.ndim == 1:
                df = df[:, np.newaxis]
        elif hasattr(self.base_estimator, "predict_proba"):
            df = self.base_estimator.predict_proba(X)
            if n_classes == 2:
                df = df[:, 1:]
        else:
            raise RuntimeError('classifier has no decision_function or '
                               'predict_proba method.')

        idx_pos_class = np.arange(df.shape[1])

        return df, idx_pos_class

    def fit(self, X, y, sample_weight=None):
        """Calibrate the fitted model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        lb = LabelBinarizer()
        lb.fit(y[y != -1])
        Y = lb.transform(y)
        self.classes_ = lb.classes_

        df, idx_pos_class = self._preproc(X)
        if len(idx_pos_class) == 1 and Y.shape[1] > 1:
            idx_pos_class = [1]
        self.calibrators_ = []

        for k, this_df in zip(idx_pos_class, df.T):
            if self.method is None:
                calibrator = _DummyCalibration()
            elif self.method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            elif self.method == 'sigmoid':
                calibrator = _SigmoidCalibration()
            elif self.method == 'beta':
                calibrator = _BetaCalibration()
            elif self.method == 'semi':
                calibrator = _SemiCalibration()
                if Y.shape[1] > 1:
                    Y[np.sum(Y, axis=1) == 0] = -1
            else:
                raise ValueError('method should be None, "sigmoid", '
                                 '"isotonic", "beta" or "semi". Got %s.' %
                                 self.method)
            calibrator.fit(this_df, Y[:, k], sample_weight)
            self.calibrators_.append(calibrator)

        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas. Can be exact zeros.
        """
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))

        df, idx_pos_class = self._preproc(X)

        for k, this_df, calibrator in \
                zip(idx_pos_class, df.T, self.calibrators_):
            if n_classes == 2:
                k += 1
            proba[:, k] = calibrator.predict(this_df)

        # Normalize the probabilities
        if n_classes == 2:
            proba[:, 0] = 1. - proba[:, 1]
        else:
            proba /= np.sum(proba, axis=1)[:, np.newaxis]

        # XXX : for some reason all probas can be 0
        proba[np.isnan(proba)] = 1. / n_classes

        # Deal with cases where the predicted probability minimally exceeds 1.0
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-5)] = 1.0

        return proba


def _semi_calibration(df, y, sample_weight=None):
    """Semi-supervised probability Calibration (REFRAME 2016)

    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray, shape (n_samples,)
        The targets.

    sample_weight : array-like, shape = [n_samples] or None
        Sample weights. If None, then samples are equally weighted.
        Currently, this parameter is not used.

    Returns
    -------
    m, v, v_t, p, p_t, a, c, d
    m : float
        The means.

    v : float
        The variances.

    p : float
        The prior of class 1.

    value0 : float
        Negative probability, following Platt (1999).

    value1 : float
        Positive probability, following Platt (1999).

    References
    ----------
    REFRAME, "Semi-supervised calibration"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

    x_l, y_l, x_u = separate_labeled_unlabeled(df, y)
    n_l = np.alen(y_l)
    n_u = np.alen(x_u)
    classes = [0, 1]
    n_classes = 2

    prior0 = float(np.sum(y_l == 0))
    prior1 = n_l - prior0
    value0 = 1. / (prior0 + 2.)
    value1 = (prior1 + 1.) / (prior1 + 2.)

    means = np.zeros(2)
    varis = np.zeros(2)
    means_tilde = np.zeros(2)
    varis_tilde = np.zeros(2)
    for c in classes:
        means[c] = means_tilde[c] = np.mean(x_l[y_l == c])
        vs = np.var(x_l[y_l == c])
        if vs == 0.0:
            vs = 0.01
        varis[c] = varis_tilde[c] = vs
    prior = prior1 / n_l
    prior_tilde = prior

    diri = dirichlet(np.bincount(y_l.astype(int)))

    def objective(params):
        tiny = np.finfo(np.float).tiny
        m, m_t, v, v_t, p, p_t, a = preproc_params(params)
        l_l = np.zeros((n_l, n_classes))
        l_l_tilde = np.zeros((n_l, n_classes))
        l_u_tilde = np.zeros((n_u, n_classes))
        p_m = 0
        p_p = 0
        p_v = 0
        for c in classes:
            if c == 0:
                p_c = 1.0 - p
                p_t_c = 1.0 - p_t
            else:
                p_c = p
                p_t_c = p_t
            l_l[:, c] = joint_likelihood(x_l, m[c], v[c], p_c) + tiny
            l_l_tilde[:, c] = joint_likelihood(x_l, m_t[c],
                                               v_t[c], p_t_c) + tiny

            if x_u is not None:
                l_u_tilde[:, c] = joint_likelihood(x_u, m_t[c],
                                                   v_t[c], p_t_c) + tiny

            s_m = np.log(t.pdf(m_t[c] - m[c], 2.*a, 0., 2.*a))
            t_s_m = np.log(norm.pdf(m[c], means[c], 1))
            p_m += t_s_m + s_m
            s_v = np.log(t.pdf(v_t[c] - v[c], 2.*a, 0., 2.*a))
            t_s_v = np.log(gamma.pdf(1./v[c], 2))
            p_v += t_s_v + s_v
            soft = np.log(t.pdf(p_t_c - p_c, 2.*a, 0, 2.*a))
            p_p += np.log(diri.pdf([p_c])) + soft
        l_l += tiny
        l_l_tilde += tiny
        l_u_tilde += tiny
        p_l = l_l / np.sum(l_l, axis=1).reshape(-1, 1)
        p_l = p_l[np.arange(n_l), y_l.astype(int)]
        log_L = p_v + p_m + p_p + np.sum(np.log(p_l))
        log_L += np.sum(np.log(l_l_tilde))
        if x_u is not None:
            log_L += np.sum(np.log(l_u_tilde))
        return -log_L

    params0 = np.hstack([means.flatten(), means_tilde.flatten(),
                         varis.flatten(), varis_tilde.flatten(),
                         prior, prior_tilde, 5.0])
    nfc = 4
    bounds = []
    bounds.extend([(-np.inf, np.inf) for i in np.arange(nfc)])
    bounds.extend([(0.01, 10) for i in np.arange(nfc)])
    bounds.extend([(0.01, 0.99), (0.01, 0.99)])
    bounds.append((0.1, 50))

    params = fmin_l_bfgs_b(objective, params0, approx_grad=True, disp=False,
                           bounds=bounds)[0]

    m, m_t, v, v_t, p, p_t, a = preproc_params(params)
    return m, v, p, value0, value1


class _SemiCalibration(BaseEstimator, RegressorMixin):
    """Semi-supervised calibration model.

    Attributes
    ----------
    m_ : float
        The means.

    v_ : float
        The variances.

    p_ : float
        The prior of class 1.

    value0_ : float
        Negative probability, following Platt (1999).

    value1_ : float
        Positive probability, following Platt (1999).
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Training data.

        y : array-like, shape (n_samples,)
            Training target.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        self.m_, self.v_, self.p_, self.value0_, self.value1_ = \
                                                       _semi_calibration(X, y,
                                                             sample_weight)
        return self

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : array, shape (n_samples,)
            The predicted data.
        """
        T = column_or_1d(T)
        a = self.value1_ - self.value0_
        b = self.value0_

        n = len(T)
        n_classes = 2
        l = np.zeros((n, n_classes))
        for c in np.arange(n_classes):
            if c == 0:
                prior = 1.0 - self.p_
            else:
                prior = self.p_
            mean = self.m_[c]
            variance = self.v_[c]
            l[:, c] = joint_likelihood(T, mean, variance, prior)
        p = l / np.sum(l, axis=1).reshape(-1, 1)
        return a * p[:, 1] + b


def separate_labeled_unlabeled(X, y):
    if -1 in y:
        x_l = X[y != -1]
        y_l = y[y != -1]
        x_u = X[y == -1]
    else:
        x_l = X
        y_l = y
        x_u = None
    return x_l, y_l, x_u


def preproc_params(params):
    n_classes = 2
    means = params[:n_classes].reshape(n_classes)
    means_tilde = params[n_classes:2*n_classes].reshape(n_classes)
    varis = params[2*n_classes:3*n_classes].reshape(n_classes)
    varis_tilde = params[3*n_classes:4*n_classes].reshape(n_classes)
    prior = params[-3]
    prior_tilde = params[-2]
    a = params[-1]
    return means, means_tilde, varis, varis_tilde, prior, prior_tilde, a


def joint_likelihood(X, mean, variance, prior):
    l = norm.pdf(X, mean, variance) * prior
    return l