from __future__ import division

import numpy as np
import pymc3 as pm
import scipy as sp
from statsmodels.datasets import get_rdataset
from theano import tensor as T

class DPGaussianMixtureModel(object):
    def __init__(self, data=None, K=30, **kwargs):
        self.K = K
        if data is not None:
            self._define_model(data)
        self.kwargs = kwargs


    def _define_model(self, data, verbose=0):
        self.N = data.shape[0]
        with pm.Model(verbose=verbose) as self.model:
            self.alpha = pm.Gamma('alpha', 1., 1.)
            self.beta = pm.Beta('beta', 1., self.alpha, shape=self.K)
            self.w = pm.Deterministic('w', self.beta * T.concatenate([[1], T.extra_ops.cumprod(1 - self.beta)[:-1]]))
            self.component = pm.Categorical('component', self.w, shape=self.N)

            self.tau = pm.Gamma('tau', 1., 1., shape=self.K)
            self.lambda_ = pm.Uniform('lambda', 0, 5, shape=self.K)
            self.mu = pm.Normal('mu', 0, self.lambda_ * self.tau, shape=self.K)
            self.obs = pm.Normal('obs', self.mu[self.component], self.lambda_[self.component] * self.tau[self.component],
                            observed=data)

    def sample(self, n_samples=20000, n_burn=10000, thin=10):
        with self.model:
            step1 = pm.Metropolis(vars=[self.alpha, self.beta, self.w, self.lambda_, self.tau, self.mu, self.obs])
            step2 = pm.ElemwiseCategorical([self.component], np.arange(self.K))

            trace_ = pm.sample(n_samples, [step1, step2])

        self.trace = trace_[n_burn::thin]
        return self.trace

    def n_components_used(self):
        return np.apply_along_axis(lambda x: np.unique(x).size, 1, self.trace['component'])

    def post_pdfs(self, x):
        post_pdf_contribs = sp.stats.norm.pdf(np.atleast_3d(x),
                                              self.trace['mu'][:, np.newaxis, :],
                                              1. / np.sqrt(self.trace['lambda']
                                                    * self.trace['tau'])[:, np.newaxis, :])
        return (self.trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)

    def post_pdf(self, x):
        post_pdfs = self.post_pdfs(x)
        return post_pdfs.mean(axis=0)

    def mean_contribs(self, x):
        post_pdf_contribs = sp.stats.norm.pdf(np.atleast_3d(x),
                                              self.trace['mu'][:, np.newaxis, :],
                                              1. / np.sqrt(self.trace['lambda']
                                                    * self.trace['tau'])[:, np.newaxis, :])
        return (self.trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0)

    """Scikit learn methods"""
    def fit(self, data):
        self._define_model(data)
        self.sample(**self.kwargs)

    def score_samples(self, x):
        return self.post_pdf(x.flatten()).flatten()
