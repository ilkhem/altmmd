import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from distributions import GaussianKernel1D, SumOfGaussians1D, energy1D, GaussianKernel, SumOfGaussians, energy


def experiment1(n=1000, m=100, mus=None, ss=None, s=1.0):
    """
    First experiment for F1 in 1D with sum of gaussians as target, and gaussian kernel.
    An adaptive learning rate greatly improves the performance of the model.
    The experiment takes the parameter of the target dist and the kernel as input, and plots the histogram with and
    without an adaptive lr, comparing it to the initial random histogram, and the true distribution.
    """
    if mus is None:
        mus = np.array([-5, 1, 4])
    if ss is None:
        ss = np.array([1, 0.3, 2])
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
    nsteps = 2000
    z = k.sample((n, m))
    e = [energy1D(xi, p, k, z)]
    e2 = [energy1D(xi, p, k, z)]

    x = xi.copy()
    x2 = xi.copy()
    for _ in tqdm(range(nsteps)):
        z = k.sample((n, m))  # generate for each position m samples from the kernel k
        dx = -2 / (n * m) * np.sum(p.b(np.expand_dims(x, 1) + z), axis=1) + 1 / (n * (n - 1)) * (
            np.sum(k.b(np.expand_dims(x, 1) - np.expand_dims(x, 0)) - k.b(np.expand_dims(x, 0) - np.expand_dims(x, 1)),
                   axis=1))
        dx2 = -2 / (n * m) * np.sum(p.b(np.expand_dims(x2, 1) + z), axis=1) + 1 / (n * (n - 1)) * (np.sum(
            k.b(np.expand_dims(x2, 1) - np.expand_dims(x2, 0)) - k.b(np.expand_dims(x2, 0) - np.expand_dims(x2, 1)),
            axis=1))
        x -= lr * dx
        x2 -= dx2 / (p(x2) + eps)
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
    eps = 0.01
    d = 2  # dimension of the data

    # define a kernel
    s = 1
    k = GaussianKernel(s)

    # define a terget distribution
    mus = np.array([[-3, -3], [1, 1], [2, -2]])  # 2D points
    ss = np.array([0.8, 1, 2])
    p = SumOfGaussians(mus, ss)

    # start by random n points between -8 and 8
    # np.random.seed(30082018)  # fix the seed for consistent data over runs
    n = 1000
    xi = np.random.uniform(-6, 6, (n, d))

    # define a learning rate, and do the gradient descent
    m = 100
    lr = 10
    nsteps = 2000
    z = k.sample((n, m, d))
    e = [energy(xi, p, k, z)]
    e2 = [energy(xi, p, k, z)]

    x = xi.copy()
    x2 = xi.copy()
    for _ in tqdm(range(nsteps)):
        z = k.sample((n, m, d))  # generate for each position m samples from the kernel k
        dx = -2 / (n * m) * np.sum(p.b(np.expand_dims(x, 1) + z), axis=1) + 1 / (n * (n - 1)) * (
            np.sum(k.b(np.expand_dims(x, 1) - np.expand_dims(x, 0)) - k.b(np.expand_dims(x, 0) - np.expand_dims(x, 1)),
                   axis=1))
        dx2 = -2 / (n * m) * np.sum(p.b(np.expand_dims(x2, 1) + z), axis=1) + 1 / (n * (n - 1)) * (np.sum(
            k.b(np.expand_dims(x2, 1) - np.expand_dims(x2, 0)) - k.b(np.expand_dims(x2, 0) - np.expand_dims(x2, 1)),
            axis=1))
        x -= lr * dx
        x2 -= dx2 / (np.expand_dims(p(x2), 1) + eps)
        e.append(energy(x, p, k, z))
        e2.append(energy(x2, p, k, z))

    fig2, ax = plt.subplots()
    ax.plot(e, color='red')
    ax.plot(e2, color='green')
    plt.show()

if __name__ == '__main__':
    # experiment1(s=0.1)  # 1ST TEST FOR F1
    experiment2()
