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


class GaussianKernel1D(Distribution):
    """1D Gaussian Kernel in a vectorized manner"""

    def __init__(self, s=1):
        self.s = s

    def f(self, x: np.ndarray):
        return (1 / np.sqrt(2 * np.pi * self.s ** 2)) * np.exp(-(x ** 2) / (2 * self.s ** 2))

    def b(self, x: np.ndarray):
        return (-1 / self.s ** 2) * np.multiply(self.f(x), x)

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


class SumOfGaussians1D(Distribution):
    """1D sum of Gaussians in a vectorized manner"""

    def __init__(self, mus, ss):
        assert len(ss) == len(mus)
        self.a = len(mus)
        self.mus = mus
        self.ss = ss

    def _f(self, mu, s, x):
        return (1 / np.sqrt(2 * np.pi * s ** 2)) * np.exp(-(x - mu) ** 2 / (2 * s ** 2))

    def f(self, x: np.ndarray):
        return (1 / self.a) * np.sum([self._f(self.mus[i], self.ss[i], x) for i in range(self.a)], axis=0)

    def b(self, x: np.ndarray):
        return (1 / self.a) * np.sum(
            [-(1 / self.ss[i] ** 2) * np.multiply(self._f(self.mus[i], self.ss[i], x), (x - self.mus[i])) for i in
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


def energy1D(x: np.ndarray, p: SumOfGaussians1D, k: GaussianKernel1D, z: np.ndarray):
    n = x.shape[0]
    m = z.shape[1]
    return -2 / (n * m) * np.sum(p(x.reshape((n, 1)) + z)) + 1 / (n * (n - 1)) * np.sum(
        k(x.reshape((n, 1)) - x.reshape((1, n))) - n * k(0))
