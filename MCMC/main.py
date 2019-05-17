import pdb
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymc3.stats import autocorr
from scipy.stats import norm
from scipy.stats import multivariate_normal

sns.set_style('white')
np.random.seed(123)


def black_box_dist(x, ndims):
    """
    Distribution we want to infer
    Right now, this is a multivariate normal distribution centered at 0.
    """
    mean = np.array([0]*ndims)
    return multivariate_normal.pdf(x, mean=mean)


def sampler(
    ndims,
    start_pos=None,
    proposal_sd=0.5,
    n_iter=10000
):

"""
Metropolis-Hastings Algorithm to sample from our blackbox distribution
"""
   if start_pos is None:
        start_pos = np.array([1.0] * ndims)
    if isinstance(start_pos, np.ndarray):
        cur_pos = start_pos
    elif isinstance(start_pos, float):
        cur_pos = np.array([start_pos])
    elif isinstance(start_pos, list):
        cur_pos = np.array(start_pos)
    else:
        raise TypeError('start_pos must be array-like')

    for _ in range(n_iter):
        print(cur_pos)
        proposal = multivariate_normal(mean=cur_pos, cov=proposal_sd).rvs()

        density_current = black_box_dist(cur_pos, ndims)
        density_proposal = black_box_dist(proposal, ndims)

        p_accept = density_proposal / density_current
        accept = np.random.rand() < p_accept

        if accept.all():
            cur_pos = proposal

        yield cur_pos


ndims = 2   # dimension of probability space
samples = list(sampler(ndims))

if ndims == 2:
    sns.jointplot(
        x=np.array(samples)[:, 0],
        y=np.array(samples)[:, 1],
        s=10,
        alpha=0.3
    )
    plt.savefig('jointplot')

elif ndims == 1:
    ax = plt.subplot()
    sns.distplot(
        np.array(samples)[:, 0],
        ax=ax
    )
    _ = ax.set(title='Histogram of observed data',
               xlabel='x', ylabel='# observations')
    plt.savefig('histogram')

# fig, ax = plt.subplots()
# ax.plot(samples)
# _ = ax.set(xlabel='sample', ylabel='mu')
# plt.savefig('trace')

# fig, ax = plt.subplots()
# lags = np.arange(1, 100)
# ax.plot(lags, [autocorr(np.array(samples), l) for l in lags])
# plt.show()
