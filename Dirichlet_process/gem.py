"""Simulation of GEM distribution

- How does GEM distribution change with alpha?
"""
import numpy as np
from numpy.random import beta
import matplotlib.pyplot as plt


def get_beta_sample(alpha):
    return beta(1, alpha)


def GEM_generator(alpha, n_iter=50):
    """ Stick-breaking interpretation
    Note: The output vector contains cumulative sum of samples
    """
    rho_vec = list([0])
    normalizer = 1
    for _ in range(n_iter):
        sample = get_beta_sample(alpha)
        rho_vec.append(normalizer*sample + rho_vec[-1])
        normalizer *= (1 - sample)

    return np.array(rho_vec)


def GEM(alpha, n_iter=50):
    """ Stick-breaking interpretation
    Note: The output vector contains cumulative sum of samples
    """
    rho_vec = list()
    normalizer = 1
    for _ in range(n_iter):
        sample = get_beta_sample(alpha)
        rho_vec.append(normalizer*sample)
        normalizer *= (1 - sample)

    return np.array(rho_vec)


def plot(rho_vec):
    fig, ax = plt.subplots(figsize=(12, 1))

    ax.vlines(rho_vec, [0]*11, [1]*11)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    aa = GEM_generator(1)
    plot(aa)
