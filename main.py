import matplotlib
# matplotlib.use('Agg')
import argparse
import os
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
plt.style.use('seaborn')

from distributions import GaussianKernel1D, SumOfGaussians1D, energy1D, GaussianKernel, SumOfGaussians, energy, Green


def experiment1(n=250, m=100, mus=None, ss=None, s=1.0):
    """
    First experiment for F1 in 1D with sum of gaussians as target, and gaussian kernel.
    An adaptive learning rate greatly improves the performance of the model.
    The experiment takes the parameter of the target dist and the kernel as input, and plots the histogram with and
    without an adaptive lr, comparing it to the initial random histogram, and the true distribution.
    """
    if mus is None:
        mus = np.array([1])
    if ss is None:
        ss = np.array([1])
    eps = 0.01

    # define a gaussian kernel with scale s
    k = GaussianKernel1D(s)
    k2 = GaussianKernel1D(s)

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
    e2 = [energy1D(xi, p, k2, z)]

    x = xi.copy()
    x2 = xi.copy()
    for _ in tqdm(range(nsteps)):
        z = k.sample((n, m))  # generate for each position m samples from the kernel k
        dx = -2 / (n * m) * np.sum(p.b(np.expand_dims(x, 1) + z), axis=1) + 1 / (n * (n - 1)) * (
            np.sum(k.b(np.expand_dims(x, 1) - np.expand_dims(x, 0)) - k.b(np.expand_dims(x, 0) - np.expand_dims(x, 1)),
                   axis=1))
        dx2 = -2 / (n * m) * np.sum(p.b(np.expand_dims(x2, 1) + z), axis=1) + 1 / (n * (n - 1)) * (np.sum(
            k2.b(np.expand_dims(x2, 1) - np.expand_dims(x2, 0)) - k2.b(np.expand_dims(x2, 0) - np.expand_dims(x2, 1)),
            axis=1))
        dxp = -(mus[0] - x)*(p(x)<0.001)
        x -= dx / (p(x) + eps) + 0.01*dxp
        x2 -= dx2 / (p(x2) + eps)
        e.append(energy1D(x, p, k, z))
        e2.append(energy1D(x2, p, k2, z))

    # plotting
    t = np.linspace(-10, 10, 250)
    fig, axes = plt.subplots(nrows=3, ncols=1)
    for ax in axes:
        ax.plot(t, p(t))
    axes[0].hist(xi, 100, density=True)
    axes[0].set_title('initial')
    axes[1].hist(x, 100, density=True)
    axes[1].set_title('s = 0.1')
    axes[2].hist(x2, 100, density=True)
    axes[2].set_title('s = 3')

    fig2, ax = plt.subplots()
    ax.plot(e, color='red', label='s=0.1')
    ax.plot(e2, color='green', label='s=3')
    ax.legend()
    plt.show()


def experiment2(d=2, n=1000, m=100, nsteps=10000, mus=None, ss=None, s=0.5):
    """
    Second experiment for F1 in 2D. This one uses the same type of kernel and target distribution as experiment1. The
    implementation should support any dimension d for the data, but it is done in 2D for visualisation.
    Since experiment1 showed that an adaptive learning rate achieves the desirable results (a fixed lr doesn't converge
    to the right distribution), this experiment should implement it.
    """
    eps = 0.01
    title = '{}D for n={}, s={}, m={} nsteps={}'.format(d, n, s, m, nsteps)
    # print(title)

    # define a kernel
    k = GaussianKernel(s)

    # define a terget distribution
    # if mus is None:
    #     mus = np.array([[-5]*d, np.arange(-2, d-2).tolist(), np.arange(4-d, 4)[::-1].tolist()])
    # if ss is None:
    #     ss = np.array([1, 0.3, 2])

    if mus is None:
        mus = np.array([np.arange(-2, d-2).tolist()])
    if ss is None:
        ss = np.array([2])
    p = SumOfGaussians(mus, ss)

    # start by random n points between -8 and 8
    # np.random.seed(30082018)  # fix the seed for consistent data over runs
    xi = np.random.uniform(-10, 10, (n, d))

    # define a learning rate, and do the gradient descent
    z = k.sample((n, m, d))
    e = [energy(xi, p, k, z)]

    x = xi.copy()

    # BURNIN
    print("Initial burnin")
    for _ in tqdm(range(5000)):
        dxp = (mus[0] - x) * (np.expand_dims(p(x), 1) < 0.0001)
        x += 0.0001*dxp
    xb = x.copy()
    print("Finished burnin")
    for _ in tqdm(range(nsteps)):
        z = k.sample((n, m, d))  # generate for each position m samples from the kernel k
        dx = -2 / (n * m) * np.sum(p.b(np.expand_dims(x, 1) + z), axis=1) + 1 / (n * (n - 1)) * (
            np.sum(k.b(np.expand_dims(x, 1) - np.expand_dims(x, 0)) - k.b(np.expand_dims(x, 0) - np.expand_dims(x, 1)),
                   axis=1))
        dxp = -(mus[0] - x)*(np.expand_dims(p(x), 1)<0.001)
        x -= dx / (np.expand_dims(p(x), 1) + eps)
        # x -= dx * 100
        e.append(energy(x, p, k, z))

    t = np.linspace(-10, 10, 250)
    fig, axes = plt.subplots(nrows=1, ncols=d)
    for j in range(d):
        axes[j].plot(t, p.project(t, j))
        axes[j].hist(x[:, j], 100, density=True)
        axes[j].set_title('proj onto axis {}'.format(j + 1))
    fig_title = '{}D_{}_{}_{}_{}'.format(d, n, s, m, nsteps)
    fig.suptitle(title)

    fig3, axes3 = plt.subplots(nrows=2, ncols=d)
    for j in range(d):
        axes3[0, j].plot(t, p.project(t, j))
        axes3[1, j].plot(t, p.project(t, j))
        axes3[0, j].hist(xi[:, j], 100, density=True)
        axes3[1, j].hist(xb[:, j], 100, density=True)
        axes3[0, j].set_title('initial proj onto axis {}'.format(j + 1))
        axes3[1, j].set_title('burnin proj onto axis {}'.format(j + 1))
    fig3.suptitle('Points after burnin')

    cwd = '/nfs/nhome/live/ilyesk/projects/altmmd'

    # plt.savefig(cwd+'/tmp/'+fig_title+'.pdf')

    fig2, ax = plt.subplots()
    fig2.suptitle('{}D for n={}, s={}, m={} nsteps={} - energy'.format(d, n, s, m, nsteps))
    ax.plot(e)
    # plt.savefig(cwd+'/tmp/'+fig_title+'_e.pdf')
    plt.show()


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description='test altmmd')
        parser.add_argument('d', nargs='?', default=2, type=int, help='dimension of data, default=2')
        parser.add_argument('n', nargs='?', default=100, type=int, help='number of data points')
        parser.add_argument('s', nargs='?', default=0.5, type=float, help='scale of kernel')
        parser.add_argument('nsteps', nargs='?', default=2500, type=int, help='number of steps')
        parser.add_argument('m', nargs='?', default=100, type=int, help='number of noise points')
        parser.add_argument('-d', type=int, dest='-d', help='dimension of data, default=2')
        parser.add_argument('-n', type=int, dest='-n', help='number of data points')
        parser.add_argument('-s', type=float, dest='-s', help='scale of kernel')
        parser.add_argument('--nsteps', type=int, dest='--nsteps', help='number of steps')
        parser.add_argument('-m', dest='-m', type=int, help='number of noise points')
        args = parser.parse_args()
        return vars(args)

    args_dict = parse_args()
    d = args_dict['d']
    if args_dict['-d']:
        d = args_dict['-d']
    n = args_dict['n']
    if args_dict['-n']:
        n = args_dict['-n']
    m = args_dict['m']
    if args_dict['-m']:
        m = args_dict['-m']
    nsteps = args_dict['nsteps']
    if args_dict['--nsteps']:
        nsteps = args_dict['--nsteps']
    s = args_dict['s']
    if args_dict['-s']:
        s = args_dict['-s']

    n = 250
    d = 3
    nsteps=4000
    s = 0.5
    m = 100
    # experiment1(s=0.1)  # 1ST TEST FOR F1
    experiment2(d=d, n=n, s=s, m=m, nsteps=nsteps)
