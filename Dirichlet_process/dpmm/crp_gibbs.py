"""
Cluster Assignment Inference based on CRP and Gibbs Sampling

Let z_1, z_2, ..., z_n denote the n data points and 1, 2... denote
the underlying cluster identifiers. This script uses Gibbs sampling
to approximate the posterior p(z_j = i| z_1, z_2 ... z_n).
i.e. we run an inference algorithm on 'n' latent variables.

The conditional probabilities Pr(\Pi_N | \Pi_{N-1}, x) needed for Gibbs
sampling are derived using the "Chinese Restaurant Process (CRP)". Since this
process is independent of the order of the data, the Gibbs sampling idea
applies in a very natural way.

Note: The means of the clusters are assumed to be drawn from an independent
prior and is not connected with the CRP. We are not working with "Dirichlet
processes" here.

This example demonstrates how a marginalized version of the GEM distribution
can used for inference of cluster assignement.
"""
from matplotlib import pyplot as plt
from numpy.random import dirichlet, multivariate_normal, choice
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
from cycler import cycler


# generate the data (two clusters)
c1 = sp.random.normal(0, 1, size=30)
c2 = sp.random.normal(8, 1, size=30)
data = np.concatenate((c1, c2)).tolist()

# helper data structure
data_dict = dict()
for i, data_pt in enumerate(data):
    data_dict[i] = {'cluster_id': 1, 'data_val': data_pt}


# this is the CRP parameter CRP(N, alpha)
alpha = 0.2
# mu_0 and rho_0 is the base distribution from which we draw
# the cluster means.
mu_0 = 1
rho_0 = 0.5
# precision associated with each cluster
cluster_center_rho = 1

cluster_contents = dict(alpha=alpha)
# all data points initally assigned in cluster 1
cluster_contents[1] = data.copy()


def likelihood(data_pt, pts_in_cluster):
    """
    data_pt: the data point to which we want to assign a cluster
    pts_in_cluster: a set of points associated with a certain cluster

    This method returns the probability that data_pt belongs to the cluster.
    Note that the definition of "belonging" is a bit fuzzy since the
    mean and the standard deviation of the cluster is computed heuristically
    based on the points in it.
    """
    if isinstance(pts_in_cluster, list) or isinstance(pts_in_cluster, np.ndarray):
        # if the algorithm chose one of the existing clusters
        rho_ = rho_0 + len(pts_in_cluster)*cluster_center_rho
        mu_ = (cluster_center_rho * np.sum(pts_in_cluster) + rho_0 * mu_0) / rho_
        std_dev = 1.0 / np.sqrt(rho_) + 1.0 / np.sqrt(rho_0)
    else:
        # if the algorithm chose a new cluster
        mu_, rho_ = mu_0, rho_0
        std_dev = 1.0 / np.sqrt(rho_)

    return sp.stats.norm.pdf(data_pt, mu_, std_dev)


def normalize(vec):
    return vec / np.sum(vec)


def conditional_prob(data_pt):
    """
    For a given data point compute the probability
    of it belonging to each cluster.
    """
    prob_vec = list()
    for cluster_id in cluster_contents.keys():
        pts_in_cluster = cluster_contents[cluster_id]

        if not pts_in_cluster:
            import ipdb
            ipdb.set_trace()

        if isinstance(pts_in_cluster, list):
            if not pts_in_cluster:
                import ipdb
                ipdb.set_trace()
        if isinstance(pts_in_cluster, float):
            alpha = pts_in_cluster
            prob_vec.append(
                alpha *
                likelihood(data_pt, pts_in_cluster)
            )
        else:
            prob_vec.append(
                len(pts_in_cluster) *
                likelihood(data_pt, pts_in_cluster)
            )

    return normalize(prob_vec)

# cluster counter
cluster_cntr = 2
num_clusters = 0

"""
The following loop implements the Gibbs sampling algorithm.
"""
for n_iter in range(20):
    print(cluster_contents.keys())
    for data_pt, data_obj in data_dict.items():

        current_cluster = data_obj['cluster_id']
        data_val = data_obj['data_val']
        cluster_contents[current_cluster].remove(data_val)

        """ keeping the cluster_contents dict clean """
        del_list = list()
        for key, cluster_list in cluster_contents.items():
            if not cluster_contents[key]:
                del_list.append(key)

        for key in del_list:
            del cluster_contents[key]
        """ ############################# """

        # print(conditional_prob(data_pt))
        new_cluster = choice(
            [cluster_id for cluster_id in cluster_contents.keys()],
            p=conditional_prob(data_val)
        )

        if new_cluster == 'alpha':
            cluster_contents[cluster_cntr] = [data_val]
            data_obj['cluster_id'] = cluster_cntr
            cluster_cntr += 1
        else:
            new_cluster = int(new_cluster)
            cluster_contents[new_cluster].append(data_val)
            data_obj['cluster_id'] = new_cluster


    # Visualization aid to track progress
    if num_clusters != len(cluster_contents.keys()):
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        ax.set_prop_cycle(
            cycler('color', ['r', 'g', 'b', 'y', 'c', 'm', 'k'])
        )
        for cluster_id in cluster_contents.keys():
            if cluster_id == 'alpha':
                continue
            
            pts_in_cluster = cluster_contents[cluster_id]

            rho_ = rho_0 + len(pts_in_cluster)*cluster_center_rho
            mu_ = (cluster_center_rho * np.sum(pts_in_cluster) + rho_0 * mu_0) / rho_
            std_dev = 1.0 / np.sqrt(rho_) + 1.0 / np.sqrt(rho_0)

            # plot cluster distributions
            lw = max(5.0*len(pts_in_cluster)/ 30.0, 0.2)
            x = np.linspace(mu_ - 2*std_dev, mu_ + 2*std_dev)
            ax.plot(
                x, sp.stats.norm.pdf(x, mu_, std_dev),lw=lw, alpha=0.5, label='norm pdf')

        plt.savefig(str(n_iter))
        ax.cla()
        num_clusters = len(cluster_contents.keys())


# fig, ax = plt.subplots(2, 1, figsize=(9, 6))
# ax[0].hist(data)
# ax[1].bar(
#     range(len(cluster_contents.keys())),
#     [np.mean(cluster_contents[key]) for key in cluster_contents.keys()]
# )
# plt.show()
