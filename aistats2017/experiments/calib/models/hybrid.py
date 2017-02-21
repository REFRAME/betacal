from __future__ import division

import numpy as np
from scipy.stats import norm, dirichlet, gamma, t
from scipy.optimize import fmin_l_bfgs_b, minimize


class HybridClassifier(object):
    def fit(self, X, y):
        x_l, y_l, x_u = separate_labeled_unlabeled(X, y)
        n_l = np.alen(y_l)
        n_u = np.alen(x_u)
        classes = [0, 1]
        n_classes = 2
        n_features = x_l.shape[1]

        prior0 = float(np.sum(y_l == 0))
        prior1 = n_l - prior0
        value0 = 1. / (prior0 + 2.)
        value1 = (prior1 + 1.) / (prior1 + 2.)
        self._c = value1 - value0
        self._d = value0

        self._means = np.zeros((n_classes, n_features))
        self._vars = np.zeros((n_classes, n_features))
        self._means_tilde = np.zeros((n_classes, n_features))
        self._vars_tilde = np.zeros((n_classes, n_features))
        for c in classes:
            means = np.mean(x_l[y_l == c], axis=0)
            self._means[c, :] = self._means_tilde[c, :] = means
            varis = np.var(x_l[y_l == c], axis=0)
            varis[varis == 0] = 0.01
            self._vars[c, :] = self._vars_tilde[c, :] = varis
        self._prior = np.sum(y_l == 1) / n_l
        self._prior_tilde = self._prior

        diri = dirichlet(np.bincount(y_l.astype(int)))

        def objective(params):
            tiny = np.finfo(np.float).tiny
            m, m_t, v, v_t, p, p_t, a = preproc_params(params, n_features,
                                                       n_classes)
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

                for f in np.arange(n_features):
                    s_m = np.log(t.pdf(m_t[c, f] - m[c, f], 2.*a, 0., 2.*a))
                    t_s_m = np.log(norm.pdf(m[c, f], self._means[c, f], 1))
                    p_m += t_s_m + s_m
                    s_v = np.log(t.pdf(v_t[c, f] - v[c, f], 2.*a, 0., 2.*a))
                    t_s_v = np.log(gamma.pdf(1./v[c, f], 2))
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
            # print -log_L
            return -log_L

        params0 = np.hstack([self._means.flatten(), self._means_tilde.flatten(),
                             self._vars.flatten(), self._vars_tilde.flatten(),
                             self._prior, self._prior_tilde, 5.0])
        nfc = 2 * n_features * n_classes
        bounds = []
        bounds.extend([(-np.inf, np.inf) for i in np.arange(nfc)])
        bounds.extend([(0.01, 10) for i in np.arange(nfc)])
        bounds.extend([(0.01, 0.99), (0.01, 0.99)])
        bounds.append((0.1, 50))
        # print "------------------------------------------------------"
        params = fmin_l_bfgs_b(objective, params0, approx_grad=True, disp=False,
                               bounds=bounds)[0]
        # print "------------------------------------------------------"
        m, m_t, v, v_t, p, p_t, a = preproc_params(params, n_features,
                                                   n_classes)
        self._means = m
        self._vars = v
        self._means_tilde = m_t
        self._vars_tilde = v_t
        self._prior = p
        self._prior_tilde = p_t
        self._a = a

    def score_samples(self, X):
        n = len(X)
        n_classes = 2
        l = np.zeros((n, n_classes))
        for c in np.arange(n_classes):
            if c == 0:
                prior = 1.0 - self._prior
            else:
                prior = self._prior
            means = self._means[c]
            varis = self._vars[c]
            l[:, c] = joint_likelihood(X, means, varis, prior)
        return l

    def predict_proba(self, X):
        l = self.score_samples(X)
        p = l / np.sum(l, axis=1).reshape(-1, 1)
        return self._c * p + self._d

    def accuracy(self, X, y):
        p = self.predict_proba(X)
        predictions = np.argmax(p, axis=1)
        return np.mean(predictions == y)


def preproc_params(params, n_features, n_classes):
    nfc = n_features*n_classes
    means = params[:nfc].reshape(n_classes, n_features)
    means_tilde = params[nfc:2*nfc].reshape(n_classes, n_features)
    varis = params[2*nfc:3*nfc].reshape(n_classes, n_features)
    varis_tilde = params[3*nfc:4*nfc].reshape(n_classes, n_features)
    prior = params[-3]
    prior_tilde = params[-2]
    a = params[-1]
    return means, means_tilde, varis, varis_tilde, prior, prior_tilde, a


def joint_likelihood(X, means, variances, prior):
    l = np.ones(X.shape[0])*prior
    for i, (m, v) in enumerate(zip(means, variances)):
        l *= norm.pdf(X[:, i], m, v)
    return l


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


if __name__ == '__main__':
    np.random.seed(42)
    x0 = np.random.multivariate_normal([2.5, 3.5], [[2.5, 0], [0, 0.09]], 200)
    x1 = np.random.multivariate_normal([2.5, 2.0], [[2.5, 0], [0, 0.09]], 200)

    X = np.vstack([x0, x1])
    y = np.ones(400)*-1
    y[np.random.choice(200, 2, replace=False)] = 0
    y[np.random.choice(np.arange(200, 400), 2, replace=False)] = 1
    hf = HybridClassifier()
    hf.fit(X, y)

    t0 = np.random.multivariate_normal([2.5, 3.5], [[2.5, 0], [0, 0.09]], 100)
    t1 = np.random.multivariate_normal([2.5, 2.0], [[2.5, 0], [0, 0.09]], 100)

    x_test = np.vstack([t0, t1])
    y_test = np.ones(200)
    y_test[:100] = 0
    print hf.accuracy(x_test, y_test)
