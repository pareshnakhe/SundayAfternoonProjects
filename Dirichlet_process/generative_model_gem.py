"""
A generative mixture model with an indefinite number of components.

This script uses the GEM(α) distribution to achieve this.
"""
import numpy as np
from numpy.random import dirichlet, multivariate_normal, uniform
from gem import GEM_generator
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def get_gem_clusters(plot=True):
    # 5 is the alpha parameter
    # we limit the number of "sticks" to 20
    stick_pos = GEM_generator(5, 20)
    cluster_prob = stick_pos[1:] - stick_pos[:-1]

    if plot:
        _, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.bar(range(1, 21), cluster_prob)
        ax.set_title('GEM(5) with 20 iterations')
        ax.set_xlabel('Cluster number')
        ax.set_ylabel('Probability')
        plt.savefig('gem(5)')

    return stick_pos[1:]


def get_component():
    """Given a uniform sample in [0, 1], find the cluster that corresponds to.
    """
    uniform_sample = uniform()

    for i, stick_boundary in enumerate(stick_pos):
        if uniform_sample <= stick_boundary:
            return i


stick_pos = get_gem_clusters()

component_means = dict()
data_points = list()
cluster_id = list()

for _ in range(100):

    component = get_component()
    if component not in component_means.keys():
        component_means[component] = multivariate_normal(
            [30, 30],
            80*np.eye(2, 2)
        )

    sample = multivariate_normal(
        component_means[component],
        2*np.eye(2, 2),
        size=1
    )

    data_points.append(sample)
    cluster_id.append(component)


data_points = np.array(data_points).reshape(100, -1)

comp_means_array = np.array([mean for mean in component_means.values()])
plt.scatter(
    comp_means_array[:, 0],
    comp_means_array[:, 1],
    s=80,
    c='b',
    marker='s'
)

plt.scatter(
    data_points[:, 0],
    data_points[:, 1],
    c=cluster_id,
    cmap='viridis', alpha=0.5
)

plt.title('Generative Model using GEM Distribution')
plt.savefig('gen_model_gem')
