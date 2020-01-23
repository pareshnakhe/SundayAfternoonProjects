"""
Generative model based on mixture of DP

No changes made to source code.
https://docs.pymc.io/notebooks/dp_mix.html"""

from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd

SEED = 5132290  # from random.org

np.random.seed(SEED)

N = 5       # num of independent draws from DP(2, N(0, 1))
K = 30      # point where we truncate the distribution

alpha = 2      # DP parameter
P0 = sp.stats.norm  # base distribution


def f(x, theta): return sp.stats.norm.pdf(x, theta, 0.3)


beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
x_plot = np.linspace(-3, 3, 200)

w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)

theta = P0.rvs(size=(N, K))

dpm_pdf_components = f(
    x_plot[np.newaxis, np.newaxis, :], theta[..., np.newaxis]
)
dpm_pdfs = (w[..., np.newaxis] * dpm_pdf_components).sum(axis=1)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_plot, dpm_pdfs.T, c='gray')

ax.set_yticklabels([])
plt.show()
