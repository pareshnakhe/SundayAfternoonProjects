import numpy as np
import math
from sklearn.neighbors import KernelDensity


class KdeEstimator:

    def __init__(self):
        self.max_param_y = None
        self.kde_core = None

        # internal parameters
        self.bandwidth = 6.5

    def fit(self, observed_data):
        self.kde_core = KernelDensity(
            kernel='gaussian', bandwidth=self.bandwidth
        ).fit(observed_data)
        self.max_param_y = np.percentile(observed_data[:, 1], 95)

    def _get_norm_factor(self, param_x):
        """
        norm_factor is the inverse of the probability of a given half configuration.
        for example, if param_x = 20,
        then this function returns 1 / Pr([20, *]).
        """
        yy = np.arange(self.max_param_y)
        xx = np.full(yy.shape, fill_value=param_x)

        discretized_space = np.column_stack((xx, yy))
        # Pr([param_x, *])
        density = np.exp(
            self.kde_core.score_samples(discretized_space)
        ).sum()
        return math.pow(density, -1)

    def get_posterior(self, param_x):
        """
		Computes posterior distribution on param_y
		:return  numpy array, where ith row corresponds to [param_y_val, probability]
		"""
        norm_factor = self._get_norm_factor(param_x)

        grid_pts = np.arange(0, self.max_param_y)
        fixed_pts = np.full_like(grid_pts, param_x)
        sample_pts = np.column_stack((fixed_pts, grid_pts))
        density_sum = np.exp(
            self.kde_core.score_samples(sample_pts)
        ) * norm_factor

        return np.column_stack((sample_pts.sum(axis=1), density_sum))
