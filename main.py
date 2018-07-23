import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import multivariate_normal as mv
from tqdm import tqdm


def k(x, s=1):
    """
    Defines a multi-dimensional Gaussian kernel
    :param x: n points of dimension d to evaluate the kernel on, should be of hsape (n,d)
    :param s: covariance (or variance if a scalar) of the Gaussian kernel
    :return: evaluated kernel of shape (n,)
    """
    if np.ndim(x) > 1:
        mu = np.zeros(x.shape[-1])
    else:
        mu = 0
    return mv.pdf(x, mean=mu, cov=s)


def dk(x, s=1):
    """
    The gradient of a multi-dimensional Gaussian kernel
    :param x: n points of dimension d to evaluate the gradient on, should be of shape (n,d)
    :param s: covariance (or variance if a scalar)
    :return: evaluated gradient of shape (n,d)
    """
    return (-1 / s) * (k(x, s) * x.T).T


def p(x, mus, ss):
    """
    The target probability distribution. This one is a sum of Gaussian
    :param x: n points of dimension d to evaluate the gradient on, should be of shape (n,d)
    :param mus: means for the Gaussian
    :param ss: covariances for the Gaussians
    :return: evaluated distribution of shape (n,)
    """
    a = len(mus)
    return np.sum([mv.pdf(x, mean=mus[i], cov=ss[i]) for i in range(a)], axis=0)


def dp(x, mus, ss):
    """
    The gradient of the target probability distribution. This one is a sum of Gaussian
    :param x: n points of dimension d to evaluate the gradient on, should be of shape (n,d)
    :param mus: means for the Gaussian of shape (a,d)
    :param ss: covariances for the Gaussians of shape (a,)
    :return: evaluated gradient distribution of shape (n,d)
    """
    # TODO: extend to non circular covariances
    a = len(mus)
    return np.sum([-(1 / ss[i]) * (mv.pdf(x, mean=mus[i], cov=ss[i]) * (x - mus[i]).T).T for i in range(a)],
                  axis=0)


def energy(x, p, k):
    """
    define the energy to minimize.
    :param x: energy evaluated at x of shape (n,d)
    :param p: the target probability
    :param k: the smoothing kernel
    :return: the energy of shape (1,)
    """
    pwx = sp.spatial.distance.pdist(x[:, None], 'euclidean')
    xz = z + x[:, np.newaxis]
    return 2/(n*(n-1))*np.sum(k(pwx, s)) - 2/(n*m)*np.sum(p(xz.flatten(), mus, ss))


if __name__ == '__main__':

    s = 2
    mus = [-5,-1, 3]
    ss = [1, 0.3, 0.5]

    # start by random n points between -10 and 10
    n = 50
    x = np.random.uniform(-7, 7, n)

    # generate for each position m samples from the kernel k
    m = 100
    z = np.random.normal(0, s, (n,m))

    t = np.linspace(-10, 10, 100)
    pt = p(t, mus, ss)

    print(energy(x, p,k))

    lr = 10
    xnew = x.copy()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t, pt)
    ax.scatter(xnew, [0]*len(xnew))
    plt.draw()

    for _ in tqdm(range(2000)):
        for i in range(n):
            dxi = -2/(n*m)*np.sum(dp(xnew[i]+z[i], mus, ss)) + 2/(n*(n-1))*(np.sum(dk(xnew[i] - xnew, s) - dk(np.array([0]),s)))
            xnew[i] = xnew[i] - lr*dxi
            #print('Energy at step (%i, %i): %f' %(_,i, energy(x,p,k)))
        del ax.collections[0]
        ax.scatter(xnew, [0]*len(xnew), color='red')
        plt.draw()
        plt.pause(0.0001)

    print(energy(xnew, p,k))

    plt.waitforbuttonpress()