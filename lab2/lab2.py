import random

import pyod.utils.data
from matplotlib import pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

# data = pyod.utils.data.generate_data_clusters( n_train=1000, n_test=500, n_clusters=2, n_features=2, contamination=0.1, size='same', density='same', dist=0.25, random_state=None, return_in_clusters=False)
#
# isotropic_gaussian_cluster = make_blobs(
#     n_samples=100, n_features=2, centers=None, cluster_std=1.0,
#     center_box=(-10.0, 10.0), shuffle=True, random_state=None, return_centers=False
# )

def normal_dist(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def model(x, mean, sd):
    a = random.randrange(0, 100)
    b = random.randrange(0, 100)
    return a * x + b + normal_dist(x, mean, sd)

# "For various values of µ, σ2 generate
# data and compute the leverage scores for all the points. "
mean = np.linspace(-10, 10, 20)
sd = np.linspace(-10, 10, 20)

x_values = np.linspace(-10, 10, 20)

y_values = [model(x_val, mean_val, sd_val) for x_val in x_values for mean_val in mean for sd_val in sd]

print(y_values)

X = []

for idx, y in enumerate(y_values):
    X.append([1, y])

print(X)

print(np.array(X).shape)

X = np.array(X)

U, S, Vh = np.linalg.svd(X, full_matrices=False)

H = U @ U.T

print(np.trace(H))

# assert np.trace(H) > 1.9 and np.trace(H) < 2 // trebuie sa fie ceva foarte aproape de 2

for idx, H_idx in enumerate(H):
    print('H_{idx}'.format(idx=idx), H_idx[idx])
    # print(H_idx[idx])

# four types of points:
# 1. regular (low noise, close to the model)

