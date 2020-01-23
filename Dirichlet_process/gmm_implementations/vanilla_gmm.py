"""https://docs.pymc.io/notebooks/gaussian_mixture_model.html
"""

import theano.tensor as tt
import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set_style('darkgrid')


np.random.seed(12345)  # set random seed for reproducibility

k = 3
ndata = 1000
spread = 4
centers = np.array([-spread, 0, spread])

# simulate data from mixture distribution
v = np.random.randint(0, k, ndata)
data = centers[v] + np.random.randn(ndata)

# plt.hist(data, density=True)
# plt.show()

""" Model Definition """

model = pm.Model()
with model:
    # cluster sizes
    p = pm.Dirichlet('p', a=np.array([1., 1., 1.]), shape=k)

    # ensure all clusters have some points
    p_min_potential = pm.Potential(
        'p_min_potential', tt.switch(tt.min(p) < .1, -np.inf, 0))

    # cluster centers
    means = pm.Normal('means', mu=[0, 0, 0], sigma=15, shape=k)

    # break symmetry
    order_means_potential = pm.Potential(
        'order_means_potential',
        tt.switch(means[1]-means[0] < 0, -np.inf, 0)
        + tt.switch(means[2]-means[1] < 0, -np.inf, 0)
    )

    # measurement error
    sd = pm.Uniform('sd', lower=0, upper=20)

    # latent cluster of each observation
    category = pm.Categorical('category',
                              p=p,
                              shape=ndata)

    # likelihood for each observed value
    points = pm.Normal('obs',
                       mu=means[category],
                       sigma=sd,
                       observed=data)

""" Fit Model """

with model:
    step1 = pm.Metropolis(vars=[p, sd, means])
    step2 = pm.CategoricalGibbsMetropolis(vars=[category])
    tr = pm.sample(10000, step=[step1, step2], tune=5000)
