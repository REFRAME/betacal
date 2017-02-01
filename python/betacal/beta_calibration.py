from __future__ import division
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import indexable, column_or_1d
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


import warnings


def _beta_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

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
    return map, lr


class _BetaCal(BaseEstimator, RegressorMixin):
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

        self.map_, self.lr_ = _beta_calibration(X, y, sample_weight)

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


def _betaAM_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

    x = np.log(df / (1. - df))

    lr = LogisticRegression(C=99999999999)
    lr.fit(x, y)
    coefs = lr.coef_[0]
    inter = lr.intercept_[0]
    a = coefs[0]
    b = a
    m = 1.0 / (1.0 + np.exp(inter / a))
    map = [a, b, m]
    return map, lr


class _BetaAMCal(BaseEstimator, RegressorMixin):
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

        self.map_, self.lr_ = _beta2_calibration(X, y, sample_weight)

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


def _betaAB_calibration(df, y, sample_weight=None):
    warnings.filterwarnings("ignore")

    df = column_or_1d(df).reshape(-1, 1)
    df = np.clip(df, 1e-16, 1-1e-16)
    y = column_or_1d(y)

    x = np.hstack((df, 1. - df))
    x = np.log(2 * x)

    lr = LogisticRegression(fit_intercept=False, C=99999999999)
    lr.fit(x, y)
    coefs = lr.coef_[0]
    a = coefs[0]
    b = -coefs[1]
    m = 0.5
    map = [a, b, m]
    return map, lr


class _BetaABCal(BaseEstimator, RegressorMixin):
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

        self.map_, self.lr_ = _beta05_calibration(X, y, sample_weight)

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
