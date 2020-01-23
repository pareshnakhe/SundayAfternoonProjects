"""
This script focuses on the inference problem in a Gaussian mixture model.

A natural parameterization of the Gaussian mixture model is as the latent variable model:

\mu_1 ... \mu_K ~ N(0, \sigma^2)    # prior on cluster means
\tau_1 ... \tau_K ~ Gamma(a, b)     # prior on precision
w ~ Dirichlet(\alpha)               # mixing distribution
z|w ~ Categorical(w)                # prob of choosing component z
x | z ~ N(\mu_z, \tau_i^{-1})       # data distribution within cluster

We could model the problem exactly as described above except that mixing time can be large
and ineffective exploration of the tails of the distribution.
(See: https://docs.pymc.io/notebooks/gaussian_mixture_model.html )

Source: https://docs.pymc.io/notebooks/marginalized_gaussian_mixture_model.html
"""
from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns


SEED = 383561

np.random.seed(SEED)  # from random.org, for reproducibility
N = 1000

W = np.array([0.35, 0.4, 0.25])

MU = np.array([0., 2., 5.])
SIGMA = np.array([0.5, 0.5, 1.])

component = np.random.choice(MU.size, size=N, p=W)
x = np.random.normal(MU[component], SIGMA[component], size=N)

# fig, ax = plt.subplots(figsize=(8, 6))

# ax.hist(x, bins=30, density=True, lw=0)
# plt.show()

""" Model Definition """

with pm.Model() as model:
    w = pm.Dirichlet('w', np.ones_like(W))

    mu = pm.Normal('mu', 0., 10., shape=W.size)
    tau = pm.Gamma('tau', 1., 1., shape=W.size)

    x_obs = pm.NormalMixture('x_obs', w, mu, tau=tau, observed=x)

""" Fit Model """
with model:
    trace = pm.sample(5000, n_init=10000, tune=1000, random_seed=SEED)
