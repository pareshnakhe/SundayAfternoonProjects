""" 
This script implements a generative model based on
a Dirichlet Process

Inspired from:
https://docs.pymc.io/notebooks/dp_mix.html
"""
from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd
from gem import GEM

# SEED = 5132290  # from random.org
# np.random.seed(SEED)

K = 30          # point where we truncate the distribution
alpha = 2       # DP parameter
P0 = sp.stats.norm  # base distribution


def f(x, theta):
    return sp.stats.norm.pdf(x, theta, 0.3)


w = GEM(alpha, K)
theta = P0.rvs(size=w.shape)
x_plot = np.linspace(-3, 3, 200)

dpm_pdf_components = f(
    x_plot, theta[..., np.newaxis]
)
dpm_pdfs = (w[..., np.newaxis] * dpm_pdf_components).sum(axis=0)


fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax[0].plot(x_plot, dpm_pdfs, c='gray')
ax[0].set_yticklabels([])
ax[0].set_title('pdf of mixture model')

ax[1].bar(theta, w, width=0.01,)
ax[1].set_title('Underlying draw from DP({}, N(0, 1))'.format(alpha))
plt.show()


def plot_dpmm_density():
    """ plots the pdf of the mixture model """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x_plot, dpm_pdfs, c='gray')

    ax.set_yticklabels([])
    plt.show()


def plot_dirichlet_process_instantation():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(theta, w, width=0.01,)
    # ax.set_yticklabels([])
    plt.show()
