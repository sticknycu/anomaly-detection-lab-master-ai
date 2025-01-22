import pyod.utils.data as pyd
import pyod.models.ocsvm as ocsvm
import pyod.models.deep_svdd as deep_svdd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


X_train, X_test, Y_train, Y_test = pyd.generate_data(n_train=300, n_test=200, contamination=0.15)

def train_model(model, X_train, Y_train, X_test, Y_test):
    def tpr(tp, fn):
        return tp / (tp + fn)

    def tnr(tn, fp):
        return tn / (tn + fp)

    def balanced_accuracy(tpr, tnr):
        return (tpr + tnr) / 2

    model.fit(X_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    print(f'Balanced accuracy: {balanced_accuracy(tpr=tpr(tp, fn), tnr=tnr(tn, fp))}')

    x_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, x_pred_proba)

    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Classification')
    plt.legend()
    plt.show()

    # Function to plot 3D data
    def plot_3d(ax, X, y, title, inlier_color='blue', outlier_color='red'):
        inliers = X[y == 0]
        outliers = X[y == 1]
        ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c=inlier_color, label='Inliers', alpha=0.6)
        ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c=outlier_color, label='Outliers', alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend()

    # Ensure data is at least 3D for visualization
    X_train_3d = np.hstack([X_train, np.random.normal(size=(X_train.shape[0], 1))])
    X_test_3d = np.hstack([X_test, np.random.normal(size=(X_test.shape[0], 1))])

    # Ground truth and predictions
    Y_train_pred = model.predict(X_train)  # OCSVM predictions for training data
    Y_test_pred = model.predict(X_test)  # Predictions for test data

    # Create subplots
    fig = plt.figure(figsize=(20, 15))

    # Ground truth for training data
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d(ax1, X_train_3d, Y_train, "Training Data (Ground Truth)")

    # Ground truth for test data
    ax2 = fig.add_subplot(222, projection='3d')
    plot_3d(ax2, X_test_3d, Y_test, "Test Data (Ground Truth)")

    # Predicted labels for training data
    ax3 = fig.add_subplot(223, projection='3d')
    plot_3d(ax3, X_train_3d, Y_train_pred, "Training Data (Predicted Labels)")

    # Predicted labels for test data
    ax4 = fig.add_subplot(224, projection='3d')
    plot_3d(ax4, X_test_3d, Y_test_pred, "Test Data (Predicted Labels)")

    plt.tight_layout()
    plt.show()


# Ex 1
train_model(model=ocsvm.OCSVM(kernel='linear', contamination=0.15), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

train_model(model=ocsvm.OCSVM(kernel='rbf', contamination=0.3), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

train_model(model=deep_svdd.DeepSVDD(contamination=0.15, n_features=2), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

# Ex 2

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# Load the .mat file
cardio_data = loadmat('cardio.mat')  # Replace with the actual path

# Extract the feature matrix X and labels y
X = cardio_data['X']
y = cardio_data['y']

# Split the data: 40% for training, 60% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Display the split sizes
print("Training set shape (X, y):", X_train.shape, y_train.shape)
print("Testing set shape (X, y):", X_test.shape, y_test.shape)

import pyod.utils.data as pyd
import pyod.models.ocsvm as ocsvm
import pyod.models.deep_svdd as deep_svdd
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

# Generate synthetic data
X_train, X_test, Y_train, Y_test = pyd.generate_data(n_train=300, n_test=200, contamination=0.15)

def train_model(model, X_train, Y_train, X_test, Y_test):
    def tpr(tp, fn):
        return tp / (tp + fn)

    def tnr(tn, fp):
        return tn / (tn + fp)

    def balanced_accuracy(tpr, tnr):
        return (tpr + tnr) / 2

    model.fit(X_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    print(f'Balanced accuracy: {balanced_accuracy(tpr=tpr(tp, fn), tnr=tnr(tn, fp))}')

    x_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test, x_pred_proba)

    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Classification')
    plt.legend()
    plt.show()

    # Function to plot 3D data
    def plot_3d(ax, X, y, title, inlier_color='blue', outlier_color='red'):
        inliers = X[y == 0]
        outliers = X[y == 1]
        ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c=inlier_color, label='Inliers', alpha=0.6)
        ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c=outlier_color, label='Outliers', alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.legend()

    # Ensure data is at least 3D for visualization
    X_train_3d = np.hstack([X_train, np.random.normal(size=(X_train.shape[0], 1))])
    X_test_3d = np.hstack([X_test, np.random.normal(size=(X_test.shape[0], 1))])

    # Ground truth and predictions
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    # Create subplots
    fig = plt.figure(figsize=(20, 15))

    # Ground truth for training data
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d(ax1, X_train_3d, Y_train, "Training Data (Ground Truth)")

    # Ground truth for test data
    ax2 = fig.add_subplot(222, projection='3d')
    plot_3d(ax2, X_test_3d, Y_test, "Test Data (Ground Truth)")

    # Predicted labels for training data
    ax3 = fig.add_subplot(223, projection='3d')
    plot_3d(ax3, X_train_3d, Y_train_pred, "Training Data (Predicted Labels)")

    # Predicted labels for test data
    ax4 = fig.add_subplot(224, projection='3d')
    plot_3d(ax4, X_test_3d, Y_test_pred, "Test Data (Predicted Labels)")

    plt.tight_layout()
    plt.show()

# Ex 1
train_model(model=ocsvm.OCSVM(kernel='linear', contamination=0.15), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

train_model(model=ocsvm.OCSVM(kernel='rbf', contamination=0.3), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

train_model(model=deep_svdd.DeepSVDD(contamination=0.15, n_features=2), X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)

# Ex 2: Cross-Validation Grid Search for OneClassSVM
param_grid = {
    'ocsvm__kernel': ['linear', 'rbf', 'sigmoid'],
    'ocsvm__gamma': ['scale', 'auto', 0.1, 0.01],
    'ocsvm__nu': [0.1, 0.15, 0.2, 0.3]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ocsvm', OneClassSVM())
])

grid_search = GridSearchCV(pipeline, param_grid, scoring='balanced_accuracy', cv=3)
grid_search.fit(X_train, Y_train.ravel() * 2 - 1)  # Convert pyod labels to sklearn format

print("Best Parameters:", grid_search.best_params_)

# Test with the best parameters
best_model = grid_search.best_estimator_
Y_test_sklearn = Y_test.ravel() * 2 - 1  # Convert pyod labels to sklearn format
Y_pred = best_model.predict(X_test)
Y_pred_pyod = (Y_pred + 1) // 2  # Convert sklearn labels to pyod format

balanced_acc = np.mean((Y_pred_pyod == Y_test).ravel())
print("Balanced Accuracy on Test Set:", balanced_acc)

# Ex 3: Shuttle Dataset
shuttle_data = loadmat('shuttle.mat')
X_shuttle = shuttle_data['X']
y_shuttle = shuttle_data['y']

X_train_shuttle, X_test_shuttle, y_train_shuttle, y_test_shuttle = train_test_split(X_shuttle, y_shuttle, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_shuttle = scaler.fit_transform(X_train_shuttle)
X_test_shuttle = scaler.transform(X_test_shuttle)

# Fit OCSVM
ocsvm_model = ocsvm.OCSVM(kernel='rbf', contamination=0.1)
ocsvm_model.fit(X_train_shuttle)

# Balanced Accuracy and AUC
y_pred_ocsvm = ocsvm_model.predict(X_test_shuttle)
y_pred_proba_ocsvm = ocsvm_model.decision_function(X_test_shuttle)
roc_auc_ocsvm = roc_auc_score(y_test_shuttle, y_pred_proba_ocsvm)
print("OCSVM Balanced Accuracy:", balanced_acc)
print("OCSVM ROC AUC:", roc_auc_ocsvm)

# Fit DeepSVDD
deep_svdd_model = deep_svdd.DeepSVDD(contamination=0.1, n_features=X_train_shuttle.shape[1])
deep_svdd_model.fit(X_train_shuttle)

# Balanced Accuracy and AUC
y_pred_deepsvdd = deep_svdd_model.predict(X_test_shuttle)
y_pred_proba_deepsvdd = deep_svdd_model.decision_function(X_test_shuttle)
roc_auc_deepsvdd = roc_auc_score(y_test_shuttle, y_pred_proba_deepsvdd)
print("DeepSVDD Balanced Accuracy:", balanced_acc)
print("DeepSVDD ROC AUC:", roc_auc_deepsvdd)
