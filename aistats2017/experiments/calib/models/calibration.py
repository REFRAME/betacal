from __future__ import division
import warnings

from math import log
import numpy as np

from scipy.optimize import fmin_bfgs

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, check_array, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.fixes import signature
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import check_cv
from sklearn.metrics.classification import _check_binary_probabilistic_predictions

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GMM
from dpgmm import DPGaussianMixtureModel

from beta_calibration import _BetaCalibration
from beta_calibration import _Beta2Calibration
from beta_calibration import _BetaBinomialCalibration

from gp import myGPC
from beta_calibration import _Beta05Calibration


class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
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
        Use sigmoids (Platt's calibration) in this case.

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
    def __init__(self, base_estimator=None, method=None, cv=3,
                 score_type=None):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.score_type = score_type

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
        lb = LabelBinarizer().fit(y)
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
                base_estimator, method=self.method, score_type=self.score_type)
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
                warnings.warn("%s does not support sample_weight. Samples"
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
                    this_estimator, method=self.method,
                    score_type=self.score_type)
                if sample_weight is not None:
                    calibrated_classifier.fit(X[test], y[test],
                                              sample_weight[test])
                else:
                    calibrated_classifier.fit(X[test], y[test])
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
        # classfiers
        mean_proba = np.zeros((X.shape[0], len(self.classes_)))
        for calibrated_classifier in self.calibrated_classifiers_:
            proba = calibrated_classifier.predict_proba(X)
            mean_proba += proba

        mean_proba /= len(self.calibrated_classifiers_)

        return mean_proba

    def calibrate_scores(self, df):
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
        # Compute the arithmetic mean of the predictions of the calibrated
        # classifiers
        df = df.reshape(-1, 1)
        mean_proba = np.zeros((len(df), len(self.classes_)))
        for calibrated_classifier in self.calibrated_classifiers_:
            proba = calibrated_classifier.calibrate_scores(df)
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
    """Probability calibration with isotonic regression or sigmoid.

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

    method : 'sigmoid' | 'isotonic'
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
    def __init__(self, base_estimator, method='sigmoid',
                 score_type=None):
        self.base_estimator = base_estimator
        self.method = method
        self.score_type = score_type

    def _preproc(self, X):
        n_classes = len(self.classes_)
        if self.score_type is None:
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
        else:
            if hasattr(self.base_estimator, self.score_type):
                df = getattr(self.base_estimator, self.score_type)(X)
                if self.score_type == "decision_function":
                    if df.ndim == 1:
                        df = df[:, np.newaxis]
                elif self.score_type == "predict_proba":
                    if n_classes == 2:
                        df = df[:, 1:]
            else:
                raise RuntimeError('classifier has no ' + self.score_type
                                   + 'method.')

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
        Y = lb.fit_transform(y)
        self.classes_ = lb.classes_

        df, idx_pos_class = self._preproc(X)
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
            elif self.method == 'beta2':
                calibrator = _Beta2Calibration()
            elif self.method == 'betabinomial':
                calibrator = _BetaBinomialCalibration()
            elif self.method == 'beta05':
                calibrator = _Beta05Calibration()
            elif self.method == 'gmm':
                calibrator = _GMMCalibration()
            elif self.method == 'kde':
                calibrator = _KDECalibration()
            elif self.method == 'gpc':
                calibrator = _GaussianProcessCalibration()
            else:
                raise ValueError('method should be None, "sigmoid", '
                                 '"isotonic", "beta", "beta2", "beta05",'
                                 '"betabinomial", "gmm", '
                                 'or "kde". Got %s.' % self.method)
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

    def calibrate_scores(self, df):
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
        proba = np.zeros((len(df), n_classes))
        idx_pos_class = [0]

        for k, this_df, calibrator in \
                zip(idx_pos_class, df.T, self.calibrators_):
            if n_classes == 2:
                k += 1
            pro = calibrator.predict(this_df)
            if np.any(np.isnan(pro)):
                pro[np.isnan(pro)] = calibrator.predict(this_df[np.isnan(
                    pro)] + 1e-300)
            proba[:, k] = pro

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


def _sigmoid_calibration(df, y, sample_weight=None):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray, shape (n_samples,)
        The targets.

    sample_weight : array-like, shape = [n_samples] or None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    a : float
        The slope.

    b : float
        The intercept.

    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    df = column_or_1d(df)
    y = column_or_1d(y)

    F = df  # F follows Platt's notations
    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    T = np.zeros(y.shape)
    T[y > 0] = 1#(prior1 + 1.) / (prior1 + 2.)
    T[y <= 0] = 0#1. / (prior0 + 2.)
    T1 = 1. - T

    def objective(AB):
        # From Platt (beginning of Section 2.2)
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        l = -(T * np.log(P + tiny) + T1 * np.log(1. - P + tiny))
        if sample_weight is not None:
            return (sample_weight * l).sum()
        else:
            return l.sum()

    def grad(AB):
        # gradient of the objective function
        E = np.exp(AB[0] * F + AB[1])
        P = 1. / (1. + E)
        TEP_minus_T1P = P * (T * E - T1)
        if sample_weight is not None:
            TEP_minus_T1P *= sample_weight
        dA = np.dot(TEP_minus_T1P, F)
        dB = np.sum(TEP_minus_T1P)
        return np.array([dA, dB])

    AB0 = np.array([0., log((prior0 + 1.) / (prior1 + 1.))])
    AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    return AB_[0], AB_[1]


class _SigmoidCalibration(BaseEstimator, RegressorMixin):
    """Sigmoid regression model.

    Attributes
    ----------
    a_ : float
        The slope.

    b_ : float
        The intercept.
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
        self.lr = LogisticRegression(C=99999999999)
        self.lr.fit(X.reshape(-1, 1), y)
        # self.a_, self.b_ = _sigmoid_calibration(X, y, sample_weight)
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
        # return 1. / (1. + np.exp(self.a_ * T + self.b_))
        return self.lr.predict_proba(T.reshape(-1, 1))[:, 1]


class _DummyCalibration(BaseEstimator, RegressorMixin):
    """Dummy regression model. The purpose of this class is to give
    the CalibratedClassifierCV class the option to just return the
    probabilities of the base classifier.


    """
    def fit(self, X, y, sample_weight=None):
        """Does nothing.

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
        return self

    def predict(self, T):
        """Return the probabilities of the base classifier.

        Parameters
        ----------
        T : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : array, shape (n_samples,)
            The predicted data.
        """
        return T

class _GaussianProcessCalibration(BaseEstimator, RegressorMixin):
    def fit(self, X, y, sample_weight=None):
        self.model = myGPC()
        if len(X) > 200:
            subsample_idx = np.random.choice(len(X), size=200)
            X = X[subsample_idx]
            y = y[subsample_idx]
        self.model.fit(X, y)
        return self

    def predict(self, T):
        return self.model.predict_proba(T)[:,0]


class _GMMCalibration(BaseEstimator, RegressorMixin):
    """Gaussian Mixture Model calibration.

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
        # TODO : normalize the scores to mu=0, std=1
        X = column_or_1d(X).reshape(-1,1)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        prior_1 = np.true_divide(y.sum(),len(y))
        self.priors = np.array([1-prior_1, prior_1])
        self.classes_ = np.unique(y)

        self.des = {}
        for c in self.classes_:
            n_components = 5
            if n_components > sum(y==c):
                n_components = sum(y==c)
            self.des[c] = GMM(n_components=n_components)
            #self.des[c] = DPGaussianMixtureModel(K=n_components)
            self.des[c].fit(X[y==c])

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
        # TODO : apply learned normalization
        T = column_or_1d(T).reshape(-1,1)

        de_scores = np.empty((len(T),len(self.classes_)))
        for i, de in enumerate(self.des.values()):
            if type(de) in [GMM]:
                de_scores[:,i] = np.exp(de.score(T.reshape(-1,1)))
            elif type(de) in [DPGaussianMixtureModel]:
                de_scores[:,i] = de.score_samples(T.reshape(-1,1))
            else:
                de_scores[:,i] = np.exp(de.score_samples(T.reshape(-1,1)))

        joint_proba = de_scores*self.priors
        posterior_proba = joint_proba/joint_proba.sum(axis=1).reshape(-1,1)
        return posterior_proba[:,1]


class _KDECalibration(BaseEstimator, RegressorMixin):
    """Kernel Density Estimation calibration.

    Attributes
    ----------
    kernel : float
        The slope.

    bandwidth : float
        The intercept.
    """
    def fit(self, X, y, sample_weight=None, kernel='gaussian', bandwidth='default'):
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
        self.kernel = kernel
        self.bandwidth = bandwidth

        X = column_or_1d(X).reshape(-1,1)
        y = column_or_1d(y)
        X, y = indexable(X, y)

        prior_1 = np.true_divide(y.sum(),len(y))
        self.priors = np.array([1-prior_1, prior_1])
        self.classes_ = np.unique(y)

        if bandwidth == 'default':
            s_min = np.min(X)
            s_max = np.max(X)
            self.bandwidth = (np.abs(s_max-s_min)+1)/20

        self.des = {}
        for c in self.classes_:
            self.des[c] = KernelDensity(kernel=self.kernel,
                                    bandwidth=self.bandwidth)
            self.des[c].fit(X[y==c])

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
        T = column_or_1d(T).reshape(-1,1)

        de_scores = np.empty((len(T),len(self.classes_)))
        for i, de in enumerate(self.des.values()):
            if type(de) in [GMM]:
                de_scores[:,i] = np.exp(de.score(T.reshape(-1,1)))
            elif type(de) in [DPGaussianMixtureModel]:
                de_scores[:,i] = de.score_samples(T.reshape(-1,1))
            else:
                de_scores[:,i] = np.exp(de.score_samples(T.reshape(-1,1)))

        joint_proba = de_scores*self.priors
        posterior_proba = joint_proba/joint_proba.sum(axis=1).reshape(-1,1)
        return posterior_proba[:,1]

def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
    """Compute true and predicted probabilities for a calibration curve.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.

    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.

    n_bins : int
        Number of bins. A bigger number requires more data.

    Returns
    -------
    prob_true : array, shape (n_bins,)
        The true probability in each bin (fraction of positives).

    prob_pred : array, shape (n_bins,)
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    y_true = _check_binary_probabilistic_predictions(y_true, y_prob)

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    zero = bin_total == 0
    bin_total[zero] = 2
    # nonzero = bin_total != 0

    prob_true = (bin_true / bin_total)
    prob_pred = (bin_sums / bin_total)

    return prob_true, prob_pred
