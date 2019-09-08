import pandas as pd
import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import pdb
import matplotlib.pyplot as plt
from scipy.stats import zscore

DATA_SIZE = 1000


def data_generator():
    feature_vec = list()
    noise_vec = list()

    for i in range(1, 4):
        feature_vec.append(
            np.random.normal(i*2 / 10.0, 0.1, DATA_SIZE).reshape(-1, 1)
        )
    real_part = np.concatenate(tuple((vec for vec in feature_vec)), axis=1)

    for i in range(2):
        noise_vec.append(
            np.random.randint(0, 2, size=DATA_SIZE).reshape(-1, 1)
        )
    fake_part = np.concatenate(tuple((vec for vec in noise_vec)), axis=1)

    data = np.concatenate((real_part, fake_part), axis=1)
    np.random.shuffle(data.T)
    return data


data = data_generator()
wt_list = list()


def core_computation(wt):
    center_of_mass = np.mean(data, axis=0)
    result_vec = np.multiply(data - center_of_mass, wt)

    loss = np.linalg.norm(result_vec, axis=1, ord=2).mean()

    penalty_feature_wts = abs(5 - np.sum(wt))
    # penalty = np.exp((5 - np.sum(wt))**2) * 5

    # l2_norm = np.linalg.norm(wt, ord=2)
    # l1_norm = np.sum(wt)

    return [loss, penalty_feature_wts, 0]


def loss_func(wt):
    cmpts = core_computation(wt)
    return sum(cmpts)


def callback(xk):
    wt_list.append(core_computation(xk))


x0 = np.ones(data.shape[1]) / 0.5
# x always has to be positive
print(data.shape, loss_func(x0))


res = minimize(
    loss_func, x0,
    method='L-BFGS-B',  # L-BFGS-B or TNC
    jac=grad(loss_func),
    bounds=tuple(((0.0, 3.0) for _ in range(x0.shape[0]))),
    callback=callback,
    options={'disp': True}
)
print(res.fun)

y_pos = np.arange(1, data.shape[1]+1)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.bar(y_pos, res.x[:data.shape[1]])
ax.set_ylabel('Feature Importance')
plt.subplots_adjust(bottom=0.35)
plt.show()
