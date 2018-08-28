import numpy as np
from scipy.stats import multivariate_normal as mv


class Distribution:
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def f(self, x):
        pass

    def b(self, x):
        pass


class NonStaticKernel:
    def __init__(self, s=None):
        self.s = 1

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def _set_s(self, s):
        self.s = s

    def update_s(self, s):
        self._set_s(s)

    def f(self, x, s=None):
        if s is not None:
            self._set_s(s)

        if np.ndim(x) > 1:
            mu = np.zeros(x.shape[-1])
        else:
            mu = 0
        return mv.pdf(x, mean=mu, cov=self.s)

    def b(self, x, s=None):
        if s is not None:
            self._set_s(s)
        return (-1 / self.s) * (self.f(x) * x.T).T

    def sample(self, shape, s=None):
        if s is not None:
            self._set_s(s)
        return np.random.normal(0, self.s, shape)


class GaussianKernel(Distribution):
    def __init__(self, s=1):
        self.s = s

    def f(self, x: np.ndarray):
        if np.ndim(x) > 1:
            mu = np.zeros(x.shape[-1])
        else:
            mu = 0
        return mv.pdf(x, mean=mu, cov=self.s)

    def b(self, x: np.ndarray):
        return (-1 / self.s) * (self.f(x) * x.T).T

    def sample(self, shape):
        return np.random.normal(0, self.s, shape)


class SumOfGaussians(Distribution):
    def __init__(self, mus, ss):
        assert len(ss) == len(mus)
        self.a = len(mus)
        self.mus = mus
        self.ss = ss

    def f(self, x: np.ndarray):
        return (1 / self.a) * np.sum([mv.pdf(x, mean=self.mus[i], cov=self.ss[i]) for i in range(self.a)], axis=0)

    def b(self, x: np.ndarray):
        return (1 / self.a) * np.sum(
            [-(1 / self.ss[i]) * (mv.pdf(x, mean=self.mus[i], cov=self.ss[i]) * (x - self.mus[i]).T).T for i in
             range(self.a)], axis=0)


class LogSumOfGaussians(Distribution):
    def __init__(self, mus, ss):
        assert len(ss) == len(mus)
        self.a = len(mus)
        self.mus = mus
        self.ss = ss

    def f(self, x: np.ndarray):
        return np.log(
            (1 / self.a) * np.sum([mv.pdf(x, mean=self.mus[i], cov=self.ss[i]) for i in range(self.a)], axis=0))

    def b(self, x: np.ndarray):
        tmp = (1 / self.a) * np.sum([mv.pdf(x, mean=self.mus[i], cov=self.ss[i]) for i in range(self.a)], axis=0)
        return (1 / self.a) * np.sum(
            [-(1 / self.ss[i]) * (mv.pdf(x, mean=self.mus[i], cov=self.ss[i]) * (x - self.mus[i]).T).T for i in
             range(self.a)], axis=0) / tmp[:, np.newaxis]
