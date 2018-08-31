import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from distributions import GaussianKernel1D, SumOfGaussians1D, energy1D


def experiment1(n=1000, m=100, mus=None, ss=None, s=1.0):
    """
    First experiment for F1 in 1D with sum of gaussians as target, and gaussian kernel.
    An adaptive learning rate greatly improves the performance of the model.
    The experiment takes the parameter of the target dist and the kernel as input, and plots the histogram with and
    without an adaptive lr, comparing it to the initial random histogram, and the true distribution.
    """
    if mus is None:
        mus = [-5, 1, 4]
    if ss is None:
        ss = [1, 0.3, 2]
    eps = 0.01

    # define a gaussian kernel with scale s
    k = GaussianKernel1D(s)

    # define the target distribution: sum of gaussians in dim 1
    p = SumOfGaussians1D(mus, ss)

    # start by random n points between -8 and 8
    np.random.seed(30082018)  # fix the seed for consistent data over runs
    xi = np.random.uniform(-8, 8, n)

    # define a learning rate, and do the gradient descent
    lr = 10
    nsteps = 1500
    z = k.sample((n, m))
    e = [energy1D(xi, p, k, z)]
    e2 = [energy1D(xi, p, k, z)]

    x = xi.copy()
    x2 = xi.copy()
    for _ in tqdm(range(nsteps)):
        z = k.sample((n, m))  # generate for each position m samples from the kernel k
        dx = -2 / (n * m) * np.sum(p.b(x.reshape((n, 1)) + z), axis=1) + 1 / (n * (n - 1)) * (
            np.sum(k.b(x.reshape((n, 1)) - x.reshape((1, n))) - k.b(x.reshape((1, n)) - x.reshape((n, 1))), axis=1))
        dx2 = -2 / (n * m) * np.sum(p.b(x2.reshape((n, 1)) + z), axis=1) + 1 / (n * (n - 1)) * (
            np.sum(k.b(x2.reshape((n, 1)) - x2.reshape((1, n))) - k.b(x2.reshape((1, n)) - x2.reshape((n, 1))), axis=1))
        x -= lr * dx
        x2 -= np.diag(1 / (p(x2) + eps)).dot(dx2)
        e.append(energy1D(x, p, k, z))
        e2.append(energy1D(x2, p, k, z))

    # plotting
    t = np.linspace(-10, 10, 250)
    fig, axes = plt.subplots(nrows=3, ncols=1)
    for ax in axes:
        ax.plot(t, p(t))
    axes[0].hist(xi, 100, density=True)
    axes[0].set_title('initial')
    axes[1].hist(x, 100, density=True)
    axes[1].set_title('fixed lr')
    axes[2].hist(x2, 100, density=True)
    axes[2].set_title('adaptive lr')

    fig2, ax = plt.subplots()
    ax.plot(e, color='red')
    ax.plot(e2, color='green')

    plt.show()


def experiment2():
    """
    Second experiment for F1 in 2D. This one uses the same type of kernel and target distribution as experiment1. The
    implementation should support any dimension d for the data, but it is done in 2D for visualisation.
    Since experiment1 showed that an adaptive learning rate achieves the desirable results (a fixed lr doesn't converge
    to the right distribution), this experiment should implement it.
    """
    pass


if __name__ == '__main__':
    experiment1(s=0.1)  # 1ST TEST FOR F1
