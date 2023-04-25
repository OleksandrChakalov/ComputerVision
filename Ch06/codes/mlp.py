import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, data_home='./data')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.3, random_state=42)

# Normalize the data to have zero mean and unit variance
mean = np.mean(X_train)
std = np.std(X_train)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Train a multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, alpha=1e-4, solver='sgd', random_state=42)
mlp.fit(X_train, y_train)

# Evaluate the model on the testing set
accuracy = mlp.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Visualize some of the misclassified images
y_pred = mlp.predict(X_test)
misclassified = X_test[y_pred != y_test][:25]
misclassified_labels = y_pred[y_pred != y_test][:25]
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
for i, (ax, img, label) in enumerate(zip(axes.flat, misclassified, misclassified_labels)):
    img = (img * std) + mean  # Undo the normalization
    img = np.clip(img, 0, 255).astype(np.uint8)  # Clip values outside [0, 255] and convert to uint8
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set(title=f"Predicted label: {label}", xticks=[], yticks=[])
plt.tight_layout()
plt.show()
