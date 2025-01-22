import pyod.models.iforest as iforest
import pyod.models.loda as loda
from pyod.models.dif import DIF
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

def plot_colormap(X, scores, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='viridis', s=20, edgecolor='k')
    plt.colorbar(label='Anomaly Score')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Exercise 2.1 Ex. 1
def exercise_2_1():

    X, _ = make_blobs(n_samples=500, centers=1, cluster_std=1.0, random_state=42)


    projections = np.random.normal(size=(5, 2))
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)


    scores = np.zeros(len(X))
    for proj in projections:
        projected = X @ proj
        hist, bin_edges = np.histogram(projected, bins=20, range=(-5, 5), density=True)
        bin_probs = hist / hist.sum()
        bin_indices = np.digitize(projected, bin_edges) - 1
        bin_indices[bin_indices >= len(bin_probs)] = len(bin_probs) - 1
        bin_indices[bin_indices < 0] = 0
        scores += bin_probs[bin_indices]

    scores /= len(projections)  # Average over projections


    X_test = np.random.uniform(-3, 3, size=(500, 2))
    test_scores = np.zeros(len(X_test))
    for proj in projections:
        projected = X_test @ proj
        hist, bin_edges = np.histogram(projected, bins=20, range=(-5, 5), density=True)
        bin_probs = hist / hist.sum()
        bin_indices = np.digitize(projected, bin_edges) - 1
        bin_indices[bin_indices >= len(bin_probs)] = len(bin_probs) - 1
        bin_indices[bin_indices < 0] = 0
        test_scores += bin_probs[bin_indices]

    test_scores /= len(projections)  # Average over projections

    plot_colormap(X_test, test_scores, "Anomaly Scores (Test Data, 20 bins)")

    # Experiment with different bin counts
    for bins in [10, 30, 50]:
        scores = np.zeros(len(X_test))
        for proj in projections:
            projected = X_test @ proj
            hist, bin_edges = np.histogram(projected, bins=bins, range=(-5, 5), density=True)
            bin_probs = hist / hist.sum()
            bin_indices = np.digitize(projected, bin_edges) - 1
            bin_indices[bin_indices >= len(bin_probs)] = len(bin_probs) - 1
            bin_indices[bin_indices < 0] = 0
            scores += bin_probs[bin_indices]

        scores /= len(projections)
        plot_colormap(X_test, scores, f"Anomaly Scores (Test Data, {bins} bins)")

# Exercise 2.2 Ex. 2
def exercise_2_2():

    X, _ = make_blobs(n_samples=1000, centers=[(10, 0), (0, 10)], cluster_std=1.0, random_state=42)

    iforest_model = iforest.IForest(contamination=0.02)
    iforest_model.fit(X)

    X_test = np.random.uniform(-10, 20, size=(1000, 2))
    scores_iforest = iforest_model.decision_function(X_test)
    plot_colormap(X_test, scores_iforest, "IForest Anomaly Scores")

    dif_model = DIF(contamination=0.02, hidden_neurons=[10, 10])
    dif_model.fit(X)
    scores_dif = dif_model.decision_function(X_test)
    plot_colormap(X_test, scores_dif, "DIF Anomaly Scores")

    loda_model = loda.LODA()
    loda_model.fit(X)
    scores_loda = loda_model.decision_function(X_test)
    plot_colormap(X_test, scores_loda, "LODA Anomaly Scores")

    for neurons in [[5, 5], [20, 20]]:
        dif_model = DIF(contamination=0.02, hidden_neurons=neurons)
        dif_model.fit(X)
        scores_dif = dif_model.decision_function(X_test)
        plot_colormap(X_test, scores_dif, f"DIF Anomaly Scores (Neurons {neurons})")

    for bins in [5, 15, 25]:
        loda_model = loda.LODA(n_bins=bins)
        loda_model.fit(X)
        scores_loda = loda_model.decision_function(X_test)
        plot_colormap(X_test, scores_loda, f"LODA Anomaly Scores ({bins} bins)")

exercise_2_1()
exercise_2_2()
