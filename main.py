import matplotlib.pyplot as plt
from tqdm import tqdm

from distributions import *


def compute_grads():
    pass


if __name__ == '__main__':  # TESTS FOR F1

    # define a gaussian kernel with scale 1
    s = 1
    k = GaussianKernel(s)

    # define the target distribution: sum of gaussians in dim 1
    mus = [-5, 0, 4]
    ss = [1, 0.3, 2]  # with these settings, p has non negligeable values and grads in the interval [-5, 5]
    p = SumOfGaussians(mus, ss)

    # start by random n points between -10 and 10
    n = 50
    xi = np.random.uniform(-6, 5, n)

    # generate for each position m samples from the kernel k
    m = 100

    # plot p
    t = np.linspace(-10, 10, 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax.plot(t, p(t))
    ax2.plot(t, p(t))

    # define a learning rate, and do the gradient descent
    lr = 1
    nsteps = 1000
    x = xi.copy()
    dx = np.zeros_like(x)
    for j in tqdm(range(nsteps)):
        z = k.sample((n, m))
        for i in range(n):
            dx[i] = -2 / (n * m) * np.sum(p.b(x[i] + z[i])) + \
                    1 / (n * (n - 1)) * (np.sum(k.b(x[i] - x) - k.b(x - x[i])))
        # dx = -2 / (n * m) * np.sum(np.apply_along_axis(p.b, 1, x.reshape((n, 1)) + z), axis=1) + 1 / (n * (n - 1)) * (
        #     np.sum(k.b(x.reshape((n, 1)) - x.reshape((1, n))) - k.b(x.reshape((1, n)) - x.reshape((n, 1))), axis=1))
        x -= lr * dx

    ax.scatter(x, [0] * n, color='green')
    ax2.scatter(xi, [0] * n, color='red')
    # ax.hist(xi, 75, density=True, color='green')
    # ax.hist(x, 75, density=True, color='red')
    plt.show()
