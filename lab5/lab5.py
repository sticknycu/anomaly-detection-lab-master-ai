import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore

mean = [5, 10, 2]
cov = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]
data = np.random.multivariate_normal(mean, cov, 500)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.set_title("3D Dataset")
plt.show()

centered_data = data - np.mean(data, axis=0)

cov_matrix = np.cov(centered_data, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

plt.figure()
plt.step(range(1, len(sorted_eigenvalues) + 1), np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues), where='mid', label='Cumulative Explained Variance')
plt.bar(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues / np.sum(sorted_eigenvalues), alpha=0.7, label='Individual Variance')
plt.legend()
plt.title("Explained Variance")
plt.show()

projected_data = np.dot(centered_data, sorted_eigenvectors)

third_pc = projected_data[:, 2]
threshold_3rd_pc = np.quantile(third_pc, 0.9)
labels_3rd_pc = np.abs(third_pc) > threshold_3rd_pc

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels_3rd_pc, cmap='coolwarm')
plt.title("Outliers in 3rd Principal Component")
plt.show()

z_scores = zscore(projected_data, axis=0)
distances = np.linalg.norm(z_scores, axis=1)

threshold_multi = np.quantile(distances, 0.9)
multi_outliers = distances > threshold_multi

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=multi_outliers, cmap='coolwarm')
plt.title("Multi-dimensional Outliers")
plt.show()

from pyod.models.pca import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.datasets import fetch_openml

shuttle = fetch_openml(name="shuttle", version=1)
X, y = shuttle.data, shuttle.target
X = X.to_numpy().astype(np.float64)
y = (y != '1').astype(int)  # Convert to binary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca_model = PCA(contamination=y_train.mean())
pca_model.fit(X_train)

plt.figure()
plt.step(range(1, len(pca_model.explained_variance_ratio_) + 1), np.cumsum(pca_model.explained_variance_ratio_), where='mid', label='Cumulative Explained Variance')
plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), pca_model.explained_variance_ratio_, alpha=0.7, label='Individual Variance')
plt.legend()
plt.title("Explained Variance (PCA)")
plt.show()

y_train_pred = pca_model.predict(X_train)
y_test_pred = pca_model.predict(X_test)

train_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

print(f"Train Balanced Accuracy: {train_accuracy}")
print(f"Test Balanced Accuracy: {test_accuracy}")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
            Dense(8, activation='relu'),
            Dense(5, activation='relu'),
            Dense(3, activation='relu')
        ])
        self.decoder = Sequential([
            Dense(5, activation='relu'),
            Dense(8, activation='relu'),
            Dense(X_train.shape[1], activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(X_train, X_train,
                          epochs=100,
                          batch_size=1024,
                          validation_data=(X_test, X_test),
                          verbose=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

reconstructed = autoencoder.predict(X_train)
errors = np.mean(np.square(X_train - reconstructed), axis=1)


threshold = np.quantile(errors, 0.9)

labels = errors > threshold
balanced_acc = balanced_accuracy_score(y_train, labels)
print(f"Balanced Accuracy: {balanced_acc}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

noise_factor = 0.35
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
x_train_noisy = tf.clip_by_value(x_train_noisy, 0., 1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, 0., 1.)

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
x_train_noisy = x_train_noisy[..., np.newaxis]
x_test_noisy = x_test_noisy[..., np.newaxis]


class ConvAutoencoder(tf.keras.Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Sequential([
            Conv2D(8, (3, 3), activation='relu', strides=2, padding='same'),
            Conv2D(4, (3, 3), activation='relu', strides=2, padding='same'),
        ])
        self.decoder = Sequential([
            Conv2DTranspose(4, (3, 3), activation='relu', strides=2, padding='same'),
            Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same'),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Compile and train the autoencoder
autoencoder = ConvAutoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(x_train, x_train,
                          epochs=10,
                          batch_size=64,
                          validation_data=(x_test, x_test),
                          verbose=1)

# Reconstruction loss
reconstructed_train = autoencoder.predict(x_train)
train_loss = np.mean((x_train - reconstructed_train) ** 2, axis=(1, 2, 3))
threshold = np.mean(train_loss) + np.std(train_loss)

# Evaluate on noisy test set
reconstructed_test = autoencoder.predict(x_test_noisy)
test_loss = np.mean((x_test_noisy - reconstructed_test) ** 2, axis=(1, 2, 3))
anomalies = test_loss > threshold

# Compute accuracy for original and noisy images
original_accuracy = np.mean(test_loss <= threshold)
noisy_accuracy = np.mean(test_loss > threshold)
print(f"Accuracy (original images): {original_accuracy:.2f}")
print(f"Accuracy (noisy images): {noisy_accuracy:.2f}")

# Visualize the results
n_images = 5
plt.figure(figsize=(15, 10))

# Original images
for i in range(n_images):
    plt.subplot(4, n_images, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel("Original", size=15)

# Noisy images
for i in range(n_images):
    plt.subplot(4, n_images, i + n_images + 1)
    plt.imshow(x_test_noisy[i].squeeze(), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel("Noisy", size=15)

# Reconstructed from original
reconstructed = autoencoder.predict(x_test)
for i in range(n_images):
    plt.subplot(4, n_images, i + 2 * n_images + 1)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel("Reconstructed", size=15)

# Reconstructed from noisy
reconstructed_noisy = autoencoder.predict(x_test_noisy)
for i in range(n_images):
    plt.subplot(4, n_images, i + 3 * n_images + 1)
    plt.imshow(reconstructed_noisy[i].squeeze(), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel("Reconstructed (Noisy)", size=15)

plt.show()

# Train Denoising Autoencoder
history = autoencoder.fit(x_train_noisy, x_train,
                          epochs=10,
                          batch_size=64,
                          validation_data=(x_test_noisy, x_test),
                          verbose=1)

# Re-run the visualization code to see denoising results
for i in range(n_images):
    plt.subplot(4, n_images, i + 3 * n_images + 1)
    plt.imshow(reconstructed_noisy[i].squeeze(), cmap='gray')
    plt.axis('off')
    if i == 0: plt.ylabel("Reconstructed (Denoisy)", size=15)

plt.show()