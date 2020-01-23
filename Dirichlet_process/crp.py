"""Chinese Restaurant Process

This scripts simulated CRP and plots 3 snapshot-plots 
depicting changes in the clausters
"""
import numpy as np
from numpy.random import dirichlet, multivariate_normal, choice
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys


def normalize(urn_contents):
    urn_contents = [vals for vals in urn_contents.values()]

    sum_contents = np.sum(urn_contents)
    return np.array(urn_contents) / sum_contents


alpha = 5
"""
dict format: component_num -> # of nodes in cluster
'component 0' corresponds to the "special ball" (in Polya Urn terminology)
'alpha': weight associated with the special ball
"""
urn_contents = dict({0: alpha, 1: 1})
cluster_counter = 2

fig, ax = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
for i in range(3):
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))

plot_itr = 0

for itr in range(1, 100):
    cluster_ids = [cl_id for cl_id in urn_contents.keys()]
    ball_chosen = choice(cluster_ids, p=normalize(urn_contents))
    if ball_chosen == 0:
        # add new ball
        urn_contents[cluster_counter] = 1
        cluster_counter += 1
    else:
        urn_contents[ball_chosen] += 1

    if itr % 25 == 0:
        ax[plot_itr].bar(urn_contents.keys(), urn_contents.values())
        ax[plot_itr].set_title('CRP with {} iterations'.format(itr))
        ax[plot_itr].set_xlabel('Component number')
        ax[plot_itr].set_ylabel('# of points')
        plot_itr += 1

plt.show()
