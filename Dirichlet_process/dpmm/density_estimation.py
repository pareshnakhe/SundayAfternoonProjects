"""
This is an example of how to run inference on a truncated GEM
for density estimation problem.

source: https://docs.pymc.io/notebooks/dp_mix.html
"""
from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd

SEED = 5132290  # from random.org
np.random.seed(SEED)


old_faithful_df = pd.read_csv(pm.get_data('old_faithful.csv'))

old_faithful_df['std_waiting'] = (
    old_faithful_df.waiting - old_faithful_df.waiting.mean()) / old_faithful_df.waiting.std()

# fig, ax = plt.subplots(figsize=(8, 6))

# n_bins = 20
# ax.hist(old_faithful_df.std_waiting, bins=n_bins, lw=0, alpha=0.5)

# ax.set_xlabel('Standardized waiting time between eruptions')
# ax.set_ylabel('Number of eruptions')
# plt.show()

N = old_faithful_df.shape[0]    # num of components = num of data points
K = 30                          # truncation limit for DP


def stick_breaking(beta):
    portion_remaining = tt.concatenate(
        [[1], tt.extra_ops.cumprod(1 - beta)[:-1]]
    )

    return beta * portion_remaining


with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1., alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))

    tau = pm.Gamma('tau', 1., 1., shape=K)
    lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    mu = pm.Normal('mu', 0, tau=lambda_ * tau, shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau,
                           observed=old_faithful_df.std_waiting.values)

with model:
    trace = pm.sample(1000, random_seed=SEED, init='advi')


""" GEM(alpha) first 5 scalars """
for i in range(5):
    pm.traceplot(trace['w'][:, i])


"""
Doubts:

1. We are using a truncated version of GEM(alpha). How would we extend to an
arbitrarily large K?
2. Technically, all we used here is GEM(alpha) and not the Dirichlet process.
Note that we used an independent Normal distribution to infer about the means
of the clusters. The *base distribution* of a Dirichlet process never came in
the picture.
"""
