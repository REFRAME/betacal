from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import indexable, column_or_1d
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from sklearn.metrics.pairwise import rbf_kernel


import warnings

from sympy import *


def _beta_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    value0 = 1. / (prior0 + 2.)
    value1 = (prior1 + 1.) / (prior1 + 2.)
    c = value1 - value0
    d = value0

    x = np.hstack((df, 1. - df))
    x = np.log(x)
    x[:, 1] *= -1

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y)
    coefs = lr.coef_[0]

    if coefs[0] < 0:
        x = x[:, 1].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y)
        coefs = lr.coef_[0]
        a = 0
        b = coefs[0]
    elif coefs[1] < 0:
        x = x[:, 0].reshape(-1, 1)
        lr = LogisticRegression(C=99999999999)
        lr.fit(x, y)
        coefs = lr.coef_[0]
        a = coefs[0]
        b = 0
    else:
        a = coefs[0]
        b = coefs[1]
    inter = lr.intercept_[0]

    m = minimize_scalar(lambda(mh): np.abs(-b*np.log(1.-mh)-a*np.log(mh)-inter),
                        bounds=[0, 1], method='Bounded').x
    map = [a, b, m]
    return map, c, d, lr


def _beta_calibration_old(df, y, sample_weight=None):
    """Probability Calibration with beta method (Flach 2016)

    Parameters
    ----------
    df : ndarray, shape (n_samples,)
        The decision function or predict proba for the samples.
        Values must be in [0, 1]

    y : ndarray, shape (n_samples,)
        The targets.

    sample_weight : array-like, shape = [n_samples] or None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    a : float
        The difference between alpha_1 and alpha_0.

    b : float
        The difference between beta_0 and beta_1.

    m : float
        The midpoint.

    References
    ----------
    Flach, "Beta-calibration"
    """

    warnings.filterwarnings("ignore")

    df = column_or_1d(df)
    y = column_or_1d(y)

    tiny = np.finfo(np.float).tiny  # to avoid division by 0 warning
    F = np.clip(df, 1e-16, 1 - 1e-16)  # F follows Platt's notations

    # Bayesian priors (see Platt end of section 2.2)
    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    value0 = 1. / (prior0 + 2.)
    value1 = (prior1 + 1.) / (prior1 + 2.)
    T = np.zeros(y.shape)
    T[y > 0] = value1
    T[y <= 0] = value0
    T1 = 1. - T

    c = value1 - value0
    d = value0

    s, m, a, b, y = symbols('s m a b y')
    da = lambdify((s, m, a, b, y), diff(-(y*log(1/(1+(m**a/(1-m)**b)/(
        s**a/(1-s)**b))) + (1-y) * log(1 - 1/(1+(m**a/(1-m)**b)/(s**a/(
        1-s)**b)))), a))
    db = lambdify((s, m, a, b, y), diff(-(y*log(1/(1+(m**a/(1-m)**b)/(
        s**a/(1-s)**b))) + (1-y) * log(1 - 1/(1+(m**a/(1-m)**b)/(s**a/(
        1-s)**b)))), b))
    dm = lambdify((s, m, a, b, y), diff(-(y*log(1/(1+(m**a/(1-m)**b)/(
        s**a/(1-s)**b))) + (1-y) * log(1 - 1/(1+(m**a/(1-m)**b)/(s**a/(
        1-s)**b)))), m))

    def objective(ABM):
        # From Platt (beginning of Section 2.2)
        LR_inv_num = (ABM[2] ** ABM[0]) / ((1 - ABM[2]) ** ABM[1])
        LR_inv_den = ((F + tiny) ** ABM[0]) / ((1 - F + tiny) ** ABM[1])
        # E = np.exp(ABM[0] * F + # ABM[1])
        LR_inv = LR_inv_num / LR_inv_den
        # P = 1. / (1. + E)
        P = c * (1. / (1. + LR_inv)) + d
        l = -(T * np.log(P + tiny) + T1 * np.log(1. - P + tiny))
        if np.any(np.isnan(l.sum())):
            print l.sum()
        if sample_weight is not None:
            return (sample_weight * l).sum()
        else:
            # print l.sum()
            return l.sum()

    def grad(ABM):
        LR_inv_num = (ABM[2] ** ABM[0]) / ((1 - ABM[2]) ** ABM[1])
        LR_inv_den = ((F + tiny)**ABM[0]) / ((1 - F + tiny) ** ABM[1])
        # E = np.exp(ABM[0] * F + # ABM[1])
        LR_inv = LR_inv_num / LR_inv_den
        # P = 1. / (1. + E)
        P = c * (1. / (1. + LR_inv)) + d
        dA = 0
        dB = 0
        dM = 0
        for i, p in enumerate(P):
            dA += da(p, ABM[2], ABM[0], ABM[1], T[i])
            dB += db(p, ABM[2], ABM[0], ABM[1], T[i])
            dM += dm(p, ABM[2], ABM[0], ABM[1], T[i])
        return np.array([dA, dB, dM])
        # TEP_minus_T1P = P * (T * F - T1)
        # if sample_weight is not None:
        #     TEP_minus_T1P *= sample_weight
        # dA = np.dot(TEP_minus_T1P, F)
        # dB = np.sum(TEP_minus_T1P)
        # return np.array([dA, dB])

    # AB0 = np.array([0., np.log((prior0 + 1.) / (prior1 + 1.))])
    # a0 = b0 = 1.0
    AB0 = np.array([np.random.rand(), np.random.rand(),
                    np.random.rand()*(1.0 - 2e-8) + 1e-8])
    bounds = [(-10, 10), (-10, 10), (1e-8, 1.0 - 1e-8)]
    # res = minimize(objective, AB0, bounds=bounds)
    # ABM = res.x
    # AB_ = fmin_bfgs(objective, AB0, fprime=grad, disp=False)
    ABM = fmin_l_bfgs_b(objective, AB0, approx_grad=True, disp=False,
                        bounds=bounds)[0]
    return ABM[0], ABM[1], ABM[2], c, d


class _BetaCalibration(BaseEstimator, RegressorMixin):
    """Beta regression model.

    Attributes
    ----------
    a_ : float
        Difference between the alphas.

    b_ : float
        Difference between the betas.

    m_ : float
        Midpoint where the likelihood ratio is 1.
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

        self.map_, _, _, self.lr_ = _beta_calibration(X, y, sample_weight)

        return self

    def predict(self, S):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted data.
        """
        df = column_or_1d(S).reshape(-1, 1)
        df = np.clip(df, 1e-16, 1-1e-16)

        x = np.hstack((df, 1. - df))
        x = np.log(x)
        x[:, 1] *= -1
        if self.map_[0] == 0:
            x = x[:, 1].reshape(-1, 1)
        elif self.map_[1] == 0:
            x = x[:, 0].reshape(-1, 1)

        return self.lr_.predict_proba(x)[:, 1]
        # return self.c_ * self.lr_.predict_proba(x)[:, 1] + self.d_
        # LR_inv_num = (self.m_ ** self.a_) / ((1 - self.m_) ** self.b_)
        # LR_inv_den = (S**self.a_) / ((1 - S) ** self.b_)
        # E = np.exp(ABM[0] * F + # ABM[1])
        # LR_inv = LR_inv_num / LR_inv_den
        # P = 1. / (1. + E)
        # return self.c_ * (1. / (1. + LR_inv)) + self.d_
        # return 1. / (1. + LR_inv)


def _beta2_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    value0 = 1. / (prior0 + 2.)
    value1 = (prior1 + 1.) / (prior1 + 2.)
    c = value1 - value0
    d = value0

    x = np.log(df / (1. - df))

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y)
    coefs = lr.coef_[0]
    inter = lr.intercept_[0]
    a = coefs[0]
    b = a
    # m = fminbound(lambda(m): b*np.log(1.-m) - a*np.log(m) - inter, 0, 1)
    # m = minimize_scalar(lambda(mh): np.abs(b*np.log(1.-mh)-a*np.log(mh)-inter),
    #                     bounds=[0, 1], method='Bounded').x
    m = 1.0 / (1.0 + np.exp(inter / a))
    map = [a, b, m]
    return map, c, d, lr


class _Beta2Calibration(BaseEstimator, RegressorMixin):
    """Beta regression model.

    Attributes
    ----------
    a_ : float
        Difference between the alphas.

    b_ : float
        Difference between the betas.

    m_ : float
        Midpoint where the likelihood ratio is 1.
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

        self.map_, _, _, self.lr_ = _beta2_calibration(X, y, sample_weight)

        return self

    def predict(self, S):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted data.
        """
        df = column_or_1d(S).reshape(-1, 1)
        df = np.clip(df, 1e-16, 1-1e-16)

        x = np.log(df / (1. - df))
        return self.lr_.predict_proba(x)[:, 1]


def _beta05_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

    prior0 = float(np.sum(y <= 0))
    prior1 = y.shape[0] - prior0
    value0 = 1. / (prior0 + 2.)
    value1 = (prior1 + 1.) / (prior1 + 2.)
    c = value1 - value0
    d = value0

    x = np.hstack((df, 1. - df))
    x = np.log(2 * x)

    lr = LogisticRegression(fit_intercept=False, C=99999999999)
    lr.fit(x, y)
    coefs = lr.coef_[0]
    a = coefs[0]
    b = -coefs[1]
    m = 0.5
    map = [a, b, m]
    return map, c, d, lr


class _Beta05Calibration(BaseEstimator, RegressorMixin):
    """Beta regression model.

    Attributes
    ----------
    a_ : float
        Difference between the alphas.

    b_ : float
        Difference between the betas.

    m_ : float
        Midpoint where the likelihood ratio is 1.
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

        self.map_, _, _, self.lr_ = _beta05_calibration(X, y, sample_weight)

        return self

    def predict(self, S):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted data.
        """
        df = column_or_1d(S).reshape(-1, 1)
        df = np.clip(df, 1e-16, 1-1e-16)

        x = np.hstack((df, 1. - df))
        x = np.log(2 * x)
        return self.lr_.predict_proba(x)[:, 1]




class _BetaBinomialCalibration(BaseEstimator, RegressorMixin):
    """Beta-Binomial regression model.

    Attributes
    ----------
    a_ : float
        Difference between the alphas.

    b_ : float
        Difference between the betas.

    m_ : float
        Midpoint where the likelihood ratio is 1.
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
        self.x_train = X
        self.y_train = y
        self.a = 1.0
        self.b = 1.0
        self.kernel_function = rbf_kernel
        # The parameter gamma needs to be adjusted
        self.kernel_args = {'gamma':40.0}

        return self

    def predict(self, S):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        S : array-like, shape (n_samples,)
            Data to predict from.

        Returns
        -------
        S_ : array, shape (n_samples,)
            The predicted data.
        """
        # Kernel matrix between a grid and the real samples
        # with size (grid_size, num_samples)
        kernel_matrix = self.kernel_function(S.reshape(-1,1),self.x_train.reshape(-1,1),**self.kernel_args)

        kernel_matrix_pos = kernel_matrix[:,np.where(self.y_train)[0]]
        kernel_matrix_neg = kernel_matrix[:,np.where(1-self.y_train)[0]]

        # Number of positive m and negative l samples after kernel
        # This are the parameters of the Binomial
        m = kernel_matrix_pos.sum(axis=1)
        l = kernel_matrix_neg.sum(axis=1)

        mean_y = (m+self.a)/(m+self.a+l+self.b)

        return mean_y

