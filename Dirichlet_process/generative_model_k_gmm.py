"""
A basic generative mixture model with fixed number of components
"""
import numpy as np
from numpy.random import dirichlet, multivariate_normal, choice
import matplotlib.pyplot as plt

num_comp = 4
dir_alpha = np.ones(num_comp)

rho = dirichlet(dir_alpha)

component_means = multivariate_normal(
    [30, 30],
    80*np.eye(2, 2),
    size=num_comp
)

data_points = list()

for _ in range(100):
    component = choice(np.arange(0, num_comp), p=rho)
    sample = multivariate_normal(
        component_means[component],
        2*np.eye(2, 2),
        size=1
    )

    data_points.append(sample)


data_points = np.array(data_points).reshape(100, -1)

plt.scatter(
    data_points[:, 0],
    data_points[:, 1]
)
plt.show()
