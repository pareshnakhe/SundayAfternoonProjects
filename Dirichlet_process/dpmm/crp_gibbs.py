from matplotlib import pyplot as plt
from numpy.random import dirichlet, multivariate_normal, choice
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd


c1 = sp.random.normal(0, 1, size=30)
c2 = sp.random.normal(8, 1, size=20)
data = np.concatenate((c1, c2)).tolist()
# plt.hist(data)

data_dict = dict()
for i, data_pt in enumerate(data):
    data_dict[i] = {'cluster_id': 1, 'data_val': data_pt}


alpha = 0.1
mu_0 = 1
rho_0 = 0.5

cluster_center_rho = 1

cluster_contents = dict(alpha=alpha)
cluster_contents[1] = data


def likelihood(data_pt, pts_in_cluster):
    if isinstance(pts_in_cluster, list) or isinstance(pts_in_cluster, np.ndarray):
        rho_ = rho_0 + len(pts_in_cluster)*cluster_center_rho
        mu_ = (cluster_center_rho * np.mean(pts_in_cluster) + rho_0 * mu_0) / rho_
    else:
        mu_, rho_ = mu_0, rho_0

    return sp.stats.norm.pdf(data_pt, mu_, rho_)


def normalize(vec):
    return vec / np.sum(vec)


def conditional_prob(data_pt):
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
            prob_vec.append(
                pts_in_cluster *
                likelihood(data_pt, pts_in_cluster)
            )
        else:
            prob_vec.append(
                len(pts_in_cluster) *
                likelihood(data_pt, pts_in_cluster)
            )

    return prob_vec


cluster_cntr = 2

for n_iter in range(500):
    print(cluster_contents.keys())
    for data_pt, data_obj in data_dict.items():
        # print(data_pt)

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
            p=normalize(conditional_prob(data_val))
        )

        if new_cluster == 'alpha':
            cluster_contents[cluster_cntr] = [data_val]
            data_obj['cluster_id'] = cluster_cntr
            cluster_cntr += 1
        else:
            new_cluster = int(new_cluster)
            cluster_contents[new_cluster].append(data_val)
            data_obj['cluster_id'] = new_cluster


fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(
    range(len(cluster_contents.keys())),
    [np.mean(cluster_contents[key]) for key in cluster_contents.keys()]
)
plt.show()
