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


class GaussianKernel(Distribution):
    """Gaussian Kernel in more than 1D in a vectorized manner. For simplicity, only circular covariance is considered
    it should also work for 1D, but x should be reshaped into (n,1) (ndims should be at least 2) """

    def __init__(self, s):
        self.s = s  # s is still the standard deviation, the covariance is s^2*Id

    def f(self, x: np.ndarray):
        d = x.shape[-1]  # n data points of dimension d each (can also be a matrix n*n of d-dimensional vectors)
        return (1 / np.sqrt((2 * np.pi * self.s ** 2) ** d)) * np.exp(-np.sum(x ** 2, axis=-1) / (2 * self.s ** 2))

    def b(self, x: np.ndarray):
        return (-1 / self.s ** 2) * np.multiply(np.expand_dims(self.f(x), -1), x)

    def sample(self, shape):
        return np.random.normal(0, self.s, shape)


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


class SumOfGaussians(Distribution):
    """High dimension sum of Gaussians in a vectorized manner. Tests show that it works for 1D but everything
    needs to have a dimension axis (with value equal to 1) """

    def __init__(self, mus, ss):
        # assert ss.shape[0] == mus.shape[0]
        self.a = mus.shape[0]
        self.mus = mus
        self.ss = ss

    def _f(self, mu, s, x):
        d = x.shape[-1]
        return (1 / np.sqrt((2 * np.pi * s ** 2) ** d)) * np.exp(
            -np.sum((x - np.reshape(mu, (1,) * (x.ndim - 1) + (d,))) ** 2, axis=-1) / (2 * s ** 2))

    def f(self, x: np.ndarray):
        return (1 / self.a) * np.sum([self._f(self.mus[i], self.ss[i], x) for i in range(self.a)], axis=0)

    def b(self, x: np.ndarray):
        d = x.shape[-1]
        return (1 / self.a) * np.sum([-(1 / self.ss[i] ** 2) * np.multiply(
            np.expand_dims(self._f(self.mus[i], self.ss[i], x), - 1),
            (x - np.reshape(self.mus[i], (1,) * (x.ndim - 1) + (d,)))) for i in range(self.a)], axis=0)

    def _f_projected_single(self, mu, s, x):
        """When projected, data has dim 1 and we can use the simple expression of 1D"""
        return (1 / np.sqrt(2 * np.pi * s ** 2)) * np.exp(-(x - mu) ** 2 / (2 * s ** 2))

    def _f_projected(self, x, mus, ss):
        """"compute the complete projected forward. Means need to be projected too"""
        return (1 / self.a) * np.sum([self._f_projected_single(mus[i], ss[i], x) for i in range(self.a)], axis=0)

    def project(self, x: np.ndarray, axis):
        musp = np.take(self.mus, axis, -1)
        return self._f_projected(x, musp, self.ss)



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
    return -2 / (n * m) * np.sum(p.f(x.reshape((n, 1)) + z)) + 1 / (n * (n - 1)) * (np.sum(
        k.f(x.reshape((n, 1)) - x.reshape((1, n)))) - n * k(0))


def energy(x: np.ndarray, p: SumOfGaussians, k: GaussianKernel, z: np.ndarray):
    n = x.shape[0]
    m = z.shape[1]
    return -2 / (n * m) * np.sum(p.f(np.expand_dims(x, 1) + z)) + 1 / (n * (n - 1)) * (np.sum(
        k.f(np.expand_dims(x, 1) - np.expand_dims(x, 0))) - n * k(np.array([0])))
