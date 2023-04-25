import numpy as np
from skimage import io, color, transform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# Load the data
def load_data():
    # Load the images
    digits = np.zeros((500, 400))
    for i in range(10):
        for j in range(50):
            path = f"../Images/digit_6.png"
            digit = io.imread(path)
            if len(digit.shape) > 2 and digit.shape[2] == 4:
                digit = color.rgba2rgb(digit)
            digit = color.rgb2gray(digit)
            digit = transform.resize(digit, (20, 20))
            digit = np.reshape(digit, (1, 400))
            digits[i * 50 + j] = digit

    # Load the labels
    labels = np.zeros((500))
    for i in range(10):
        labels[i * 50: (i + 1) * 50] = i

    return digits, labels

# Preprocess the data
def preprocess_data(digits):
    # Replace NaN values with the mean value of each feature
    imputer = SimpleImputer()
    digits = imputer.fit_transform(digits)
    digits -= np.mean(digits, axis=0)
    digits /= np.std(digits, axis=0)
    return digits

# Split the data into training and testing sets
digits, labels = load_data()
digits = preprocess_data(digits)
X_train, X_test, y_train, y_test = train_test_split(digits, labels, test_size=0.3, random_state=42)

# Train the model
clf = LogisticRegression(random_state=42, max_iter=5000).fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
