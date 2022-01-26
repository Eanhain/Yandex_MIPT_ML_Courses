import pandas as pd
import numpy as np
import sys
from numpy import linalg as LA

adver_data = pd.read_csv(r'hw_marked_data/advertising.csv')
X = adver_data.drop("Sales", axis=1)
y = adver_data["Sales"]
means, stds = X.values.mean(axis=0), X.values.std(axis=0)
X = (X - means) / stds
X = np.hstack([X, np.ones(shape=X.shape[0]).reshape(200, -1)])


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    x = X[train_ind]
    y = y[train_ind]
    diff = 2 * (x @ w - y)
    grad0 = diff * x[0]
    grad1 = diff * x[1]
    grad2 = diff * x[2]
    grad3 = diff * x[3]
    return w - eta * np.array([grad0, grad1, grad2, grad3])


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    y = y.values
    weight_dist = np.inf
    w = w_init
    errors = []
    iter_num = 0
    np.random.seed(seed)

    while weight_dist > min_weight_dist and iter_num < max_iter:
        random_ind = np.random.randint(X.shape[0])
        errors.append(1 / X.shape[0] * LA.norm(X @ w[:, np.newaxis] - y[:, np.newaxis], 2) ** 2)
        sys.stdout.write('\r {} {} dist: {}'.format(errors[-1], w, weight_dist))
        w_new = stochastic_gradient_step(X, y, w, random_ind)
        weight_dist = np.sum(np.abs(w - w_new))
        w = w_new
    return w, errors


stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, np.array([0] * 4), max_iter=1e5)
