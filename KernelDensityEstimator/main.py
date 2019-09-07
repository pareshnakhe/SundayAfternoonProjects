from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from KdeEstimator import KdeEstimator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

plt.style.use('seaborn-dark')

# generate fake data
Nd = 2
np.random.seed(43)

mu = np.array([60, 40])

cov = np.random.randn(2, 2) * 10
cov = cov @ cov.T
mvn = multivariate_normal(mean=mu, cov=cov)
data = mvn.rvs(400)

# visualize the data
fig, axes = plt.subplots(1, 1, figsize=(14, 6))
axes.scatter(data[:, 0], data[:, 1], c='k', alpha=0.2, zorder=11)
axes.set_aspect('equal')
axes.set_xlabel('First dimension')
axes.set_ylabel('Second dimension')
plt.suptitle('Fake Data')
plt.show()


# KdeEstimator application
kd = KdeEstimator()
kd.fit(data[:380])

post_data = kd.get_posterior(60)
plt.plot(post_data[:, 0], post_data[:, 1])
plt.ylabel('posterior Probability')
plt.xlabel('possible values for second dimension')
plt.show()
