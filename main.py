import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import multivariate_normal as mv
from tqdm import tqdm


def k(x, s=1):
    if np.ndim(x) > 1:
        mu = np.zeros(x.shape[-1])
    else:
        mu = 0
    return mv.pdf(x, mean=mu, cov=s)


def dk(x, s=1):
    return -k(x, s) * (1 / s) * x


def p(x, mus, ss):
    m = len(mus)
    return np.sum([mv.pdf(x, mean=mus[i], cov=ss[i]) for i in range(m)], axis=0)


def dp(x, mus, ss):
    m = len(mus)
    return np.sum([-(1 / ss[i]) * (mv.pdf(x, mean=mus[i], cov=ss[i]) * (x - mus[i]).T).T for i in range(m)], axis=0)


def energy(x,p,k):
    pwx = sp.spatial.distance.pdist(x[:, None], 'euclidean')
    xz = z + x[:, np.newaxis]
    return 2/(n*(n-1))*np.sum(k(pwx, s)) - 2/(n*m)*np.sum(p(xz.flatten(), mus, ss))


s = 3
mus = [-4, 3]
ss = [1, 0.5]

# start by random n points between -10 and 10
n = 15
x = np.random.uniform(-7, 7, n)

# generate for each position m samples from the kernel k
m = 100
z = np.random.normal(0, s, (n,m))

t = np.linspace(-10,10,100)
pt = p(t, mus, ss)

# plt.figure()
# plt.plot(t, pt)
# plt.scatter(x, [0]*len(x), color='green')
# plt.title('Particles and target distribution')
# plt.show()

print(energy(x, p,k))

lr = 1
xnew = x.copy()

plt.ion()
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, pt)
ax.scatter(xnew, [0]*len(xnew))
plt.draw()

for _ in tqdm(range(2000)):
    for i in range(n):
        dxi = -2/(n*m)*np.sum(dp(xnew[i]+z[i], mus, ss)) + 2/(n*(n-1))*(np.sum(dk(xnew[i] - xnew, s) - dk(0,s)))
        xnew[i] = xnew[i] - lr*dxi
        #print('Energy at step (%i, %i): %f' %(_,i, energy(x,p,k)))
    del ax.collections[0]
    ax.scatter(xnew, [0]*len(xnew), color='red')
    plt.draw()
    plt.pause(0.01)

print(energy(xnew, p,k))

plt.waitforbuttonpress()