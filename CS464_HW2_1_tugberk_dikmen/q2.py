import numpy as np
import gzip
import matplotlib.pyplot as plt

# One-hot encoding of the labels
def one_hot_encoding(label_data):
    encoded_labels = np.zeros((label_data.size, 10))
    encoded_labels[np.arange(label_data.size), label_data] = 1
    return encoded_labels

# Function to read pixel data from the dataset and flatten it
def read_pixels(data_path, max_samples=None):
    with gzip.open(data_path, 'rb') as f:
        f.read(16)  # Skip header
        pixel_data = np.frombuffer(f.read(max_samples * 28 * 28 if max_samples else -1), np.uint8)
    normalized_pixels = pixel_data / 255.0  # Normalize
    flattened_pixels = normalized_pixels.reshape(-1, 28*28)  # Flatten
    return flattened_pixels

# Function to read label data from the dataset and apply one-hot encoding
def read_labels(data_path, max_samples=None):
    with gzip.open(data_path, 'rb') as f:
        f.read(8)  # Skip header
        label_data = np.frombuffer(f.read(max_samples if max_samples else -1), np.uint8)
    one_hot_encoding_labels = one_hot_encoding(label_data)
    return one_hot_encoding_labels

# Function to read the entire dataset
def read_dataset(max_samples=None):
    X_train = read_pixels("data/train-images-idx3-ubyte.gz", max_samples)
    y_train = read_labels("data/train-labels-idx1-ubyte.gz", max_samples)
    X_test = read_pixels("data/t10k-images-idx3-ubyte.gz", max_samples)
    y_test = read_labels("data/t10k-labels-idx1-ubyte.gz", max_samples)

    # Split the training data to create a validation dataset
    split_index = max_samples // 6 if max_samples else 10000
    X_val = X_train[:split_index]
    y_val = y_train[:split_index]
    X_train = X_train[split_index:]
    y_train = y_train[split_index:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# Softmax function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Cross-entropy loss with L2 regularization
def cross_entropy_loss(y_true, y_pred, weights, lambda_reg):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m + (lambda_reg / (2 * m)) * np.sum(weights ** 2)
    return loss

# Gradient of the loss function
def compute_gradient(X, y_true, y_pred, weights, lambda_reg):
    m = y_true.shape[0]
    gradient = (X.T @ (y_pred - y_true)) / m + (lambda_reg / m) * weights
    return gradient

def compute_confusion_matrix(true, pred, num_classes):
    """Compute a confusion matrix using numpy for two np.arrays
    true and pred.

    Args:
        true (array, shape = [n_samples]):
            True labels of the data.
        pred (array, shape = [n_samples]):
            Predictions.
        num_classes (int): Number of classes

    Returns:
        Confusion matrix (array, shape = [num_classes, num_classes]).
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        conf_matrix[t, p] += 1
    return conf_matrix

def precision_recall_fscore(conf_matrix):
    # Calculate Precision, Recall, F1 score, and F2 score for each class
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    f2_score = 5 * (precision * recall) / (4 * precision + recall)

    # Handling the case where the denominator is zero
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)
    f2_score = np.nan_to_num(f2_score)

    return precision, recall, f1_score, f2_score

# Logistic Regression Classifier
class LogisticRegressionClassifier:
    def __init__(self, learning_rate, lambda_reg, num_classes):
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.weights = None
        self.num_classes = num_classes

    def train(self, X, y, epochs, batch_size):
        n_samples, n_features = X.shape
        #self.weights = np.random.normal(0, 1, (n_features, self.num_classes))
        self.weights = np.zeros((n_features, self.num_classes))
        #self.weights = np.random.uniform(low=-1, high=1, size=(n_features, self.num_classes))

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred = softmax(X_batch @ self.weights)
                gradient = compute_gradient(X_batch, y_batch, y_pred, self.weights, self.lambda_reg)
                self.weights -= self.learning_rate * gradient

            # Optional: print out loss or accuracy here to monitor training

    def predict(self, X):
        y_pred = softmax(X @ self.weights)
        return np.argmax(y_pred, axis=1)

# Read a subset of the dataset for testing
X_train, y_train, X_val, y_val, X_test, y_test = read_dataset()  # Adjust max_samples as needed

# Initialize and train the logistic regression classifier
# Those paramater will be modified according to q2.2
lr=1e-1
reg_lambda=1e-9
class_num=10

epoc_num=100
batch_s=1

classifier = LogisticRegressionClassifier(lr, reg_lambda, class_num)
classifier.train(X_train, y_train, epoc_num, batch_s)

# Predict and evaluate on the test set
y_test_pred = classifier.predict(X_test)
test_accuracy = np.mean(np.argmax(y_test, axis=1) == y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

true_labels = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to integers
num_classes = 10  # Number of classes in MNIST
conf_matrix = compute_confusion_matrix(true_labels, y_test_pred, num_classes)

# Visualize the confusion matrix
#plt.figure(figsize=(10, 10))
#plt.matshow(conf_matrix, cmap=plt.cm.rainbow, fignum=1)
#plt.colorbar()
#plt.xlabel('Predicted Labels')
#plt.ylabel('True Labels')
#plt.title('Confusion Matrix \n')
#plt.show()

# Visualize the confusion matrix with values written in each box
plt.figure(figsize=(10, 10))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Visualize the confusion matrix with values written in each box
plt.figure(figsize=(10, 10))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()

# Annotating the confusion matrix with its values
thresh = conf_matrix.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
subtitle_text = f"LR: {lr}, Lambda: {reg_lambda}, Classes: {class_num}, Epochs: {epoc_num}, Batch Size: {batch_s}"
plt.suptitle('Confusion Matrix', fontsize=16)
plt.title(subtitle_text, fontsize=10)
plt.show()

# For Question 2.4
# Code to visualize weights (use your own weight variable, adjust its shape by yourself)
# Assuming 'classifier.weights' contains the trained weight matrix
# Visualize each weight vector as an image
for i in range(classifier.num_classes):
    weight_vector = classifier.weights[:, i].reshape(28, 28)  # Reshape to 28x28

    plt.figure()
    plt.matshow(weight_vector, cmap=plt.cm.gray, vmin=0.5*weight_vector.min(), vmax=0.5*weight_vector.max())
    plt.title(f"Class {i} Weight Image")
    plt.colorbar()
    plt.show()

# Compute the metrics using the confusion matrix
precision, recall, f1_score, f2_score = precision_recall_fscore(conf_matrix)

# Print the metrics for each class
for i in range(num_classes):
    print(f"Class {i}: Precision: {precision[i]:.2f}, Recall: {recall[i]:.2f}, F1 Score: {f1_score[i]:.2f}, F2 Score: {f2_score[i]:.2f}")