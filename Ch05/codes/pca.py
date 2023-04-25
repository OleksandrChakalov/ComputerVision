import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

# Load the data
mnist = fetch_openml('mnist_784', as_frame=False, version=1, cache=True, data_home="./data", return_X_y=True)
X_train, y_train = shuffle(mnist[0], mnist[1])
X_train = X_train[:5000]
y_train = y_train[:5000]

# Preprocess the data
X_train = X_train / 255.0

# Fit the PCA model
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Visualize the result
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train.astype(np.int), cmap='tab10')
plt.legend(*scatter.legend_elements(), title='Classes')
plt.show()
