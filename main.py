import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import multivariate_normal as mv
from tqdm import tqdm


class Distribution:
    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def f(self, x):
        pass

    def b(self, x):
        pass


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
        return (1/self.a)*np.sum([mv.pdf(x, mean=self.mus[i], cov=self.ss[i]) for i in range(self.a)], axis=0)

    def b(self, x: np.ndarray):
        return (1/self.a)*np.sum(
            [-(1 / self.ss[i]) * (mv.pdf(x, mean=self.mus[i], cov=self.ss[i]) * (x - self.mus[i]).T).T for i in
             range(self.a)], axis=0)


# def k(x, s=1):
#     """
#     Defines a multi-dimensional Gaussian kernel
#     :param x: n points of dimension d to evaluate the kernel on, should be of hsape (n,d)
#     :param s: covariance (or variance if a scalar) of the Gaussian kernel
#     :return: evaluated kernel of shape (n,)
#     """
#     if np.ndim(x) > 1:
#         mu = np.zeros(x.shape[-1])
#     else:
#         mu = 0
#     return mv.pdf(x, mean=mu, cov=s)


# def dk(x, s=1):
#     """
#     The gradient of a multi-dimensional Gaussian kernel
#     :param x: n points of dimension d to evaluate the gradient on, should be of shape (n,d)
#     :param s: covariance (or variance if a scalar)
#     :return: evaluated gradient of shape (n,d)
#     """
#     return (-1 / s) * (k(x, s) * x.T).T


# def p(x, mus, ss):
#     """
#     The target probability distribution. This one is a sum of Gaussian
#     :param x: n points of dimension d to evaluate the gradient on, should be of shape (n,d)
#     :param mus: means for the Gaussian
#     :param ss: covariances for the Gaussians
#     :return: evaluated distribution of shape (n,)
#     """
#     a = len(mus)
#     return np.sum([mv.pdf(x, mean=mus[i], cov=ss[i]) for i in range(a)], axis=0)
#
#
# def dp(x, mus, ss):
#     """
#     The gradient of the target probability distribution. This one is a sum of Gaussian
#     :param x: n points of dimension d to evaluate the gradient on, should be of shape (n,d)
#     :param mus: means for the Gaussian of shape (a,d)
#     :param ss: covariances for the Gaussians of shape (a,)
#     :return: evaluated gradient distribution of shape (n,d)
#     """
#     # TODO: extend to non circular covariances
#     a = len(mus)
#     return np.sum([-(1 / ss[i]) * (mv.pdf(x, mean=mus[i], cov=ss[i]) * (x - mus[i]).T).T for i in range(a)],
#                   axis=0)


def energy(x, z, k, p):
    """
    define the energy to minimize.
    :param x: energy evaluated at x of shape (n,d)
    :param p: the target probability
    :param k: the smoothing kernel
    :return: the energy of shape (1,)
    """
    pwx = sp.spatial.distance.pdist(x[:, None], 'euclidean')
    xz = z + x[:, np.newaxis]
    return 2 / (n * (n - 1)) * np.sum(k(pwx)) - 2 / (n * m) * np.sum(p(xz.flatten()))


if __name__ == '__main__':

    s_k = [0.1, 1, 2, 3]
    k = [GaussianKernel(s) for s in s_k]

    mus = [-5, -1, 3]
    ss = [1, 0.3, 0.5]

    p = SumOfGaussians(mus, ss)

    # start by random n points between -10 and 10
    n = 25
    x = np.random.uniform(-7, 7, n)

    # generate for each position m samples from the kernel k
    m = 100
    z = np.array([k[i].sample((n,m)) for i in range(len(k))])

    t = np.linspace(-10, 10, 100)
    pt = p(t)

    lr = 0.5
    xnew = np.array([x.copy() for _ in range(len(k))])

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for j in range(4):
        axes[j // 2, j % 2].plot(t, pt)
        axes[j // 2, j % 2].scatter(xnew[j], [0] * len(xnew[j]))
        plt.draw()
    dx = np.zeros_like(xnew)
    for _ in tqdm(range(800)):
        for i in range(n):
            for j in range(len(k)):
                dx[j, i] = -2 / (n * m) * np.sum(p.b(xnew[j, i] + z[j, i])) + 2 / (n * (n - 1)) * (
                    np.sum(k[j].b(xnew[j, i] - xnew) - k[j].b(np.array([0]))))
        xnew -= lr * dx
        for j in range(4):
            del axes[j // 2, j % 2].collections[0]
            axes[j // 2, j % 2].scatter(xnew[j], [0] * len(xnew[j]), color='red')
            plt.draw()
            plt.pause(0.00001)

    # print(energy(xnew, z1, k, p))

    plt.waitforbuttonpress()
