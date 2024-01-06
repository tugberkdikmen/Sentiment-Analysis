import numpy as np
import gzip
import matplotlib.pyplot as plt

# Function to read and flatten pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path, 'rb') as f:
        pixel_data = np.frombuffer(f.read(), np.uint8, offset=16)
    normalized_pixels = pixel_data / 255.0
    flattened_pixels = normalized_pixels.reshape(-1, 28*28)
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path, 'rb') as f:
        label_data = np.frombuffer(f.read(), np.uint8, offset=8)
    return label_data

# PCA Implementation
def pca(data, num_components=10,select=1):
    # Center the data
    mean_data = np.mean(data, axis=0)
    centered_data = data - mean_data

    # Compute the Covariance Matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Compute Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort Eigenvalues and Eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top num_components eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Project the data onto these eigenvectors
    projected_data = np.dot(centered_data, selected_eigenvectors)
    
    if(select == 1):
        projected_data = np.dot(centered_data, selected_eigenvectors)
        return projected_data, sorted_eigenvalues
    elif(select == 2):
        projected_data = np.dot(centered_data, selected_eigenvectors)
        return projected_data, sorted_eigenvalues, selected_eigenvectors
    elif(select == 3):
        return mean_data, selected_eigenvectors

def calculate_components_for_variance(eigenvalues, threshold=0.70):
    # Calculate the cumulative sum of the sorted eigenvalues
    cumulative_sum_eigenvalues = np.cumsum(eigenvalues)

    # Calculate the total variance
    total_variance = sum(eigenvalues)

    # Find the number of components needed to explain at least the given threshold (e.g., 70%) of the variance
    num_components = np.argmax(cumulative_sum_eigenvalues >= total_variance * threshold) + 1  # +1 for 1-based index

    return num_components

def min_max_scaling(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scaled_array = (array - min_val) / (max_val - min_val)
    return scaled_array

# Function to reconstruct an image using the first k principal components
def reconstruct_image(original_image, pca_components, mean_image, k):
    # Ensure pca_components is 2-dimensional
    pca_components_k = pca_components[:, :k] if pca_components.ndim > 1 else pca_components[:k]
    projected_image = np.dot(original_image - mean_image, pca_components_k)
    reconstructed_image = np.dot(projected_image, pca_components_k.T) + mean_image
    return reconstructed_image

# Main execution
if __name__ == "__main__":
    # File paths
    image_file_path = 'data/train-images-idx3-ubyte.gz'
    label_file_path = 'data/train-labels-idx1-ubyte.gz'

    # Read and preprocess the data
    images = read_pixels(image_file_path)
    labels = read_labels(label_file_path)

    # Apply PCA 1.1
    pca_data, eigenvalues = pca(images, 10)
    
    # Apply PCA 1.3
    _, _, pca_components = pca(images, 10, 2)

    # Calculate the Proportion of Variance Explained (PVE) for the first 10 components
    total_variance = sum(eigenvalues)
    pve_first_10 = sum(eigenvalues[:10]) / total_variance
    print(f"PVE for the first 10 principal components: {pve_first_10:.4f}")

    num_components_70_percent = calculate_components_for_variance(eigenvalues, 0.70)
    print(f"Number of components to explain at least 70% of the variance: {num_components_70_percent}")
    
    # Reshape and scale the first 10 principal components
    pca_images = [min_max_scaling(pc.reshape(28, 28)) for pc in pca_components.T]

    # Visualization of the principal component images
    plt.figure(figsize=(15, 6))
    for i, image in enumerate(pca_images, 1):
        plt.subplot(2, 5, i)
        plt.imshow(image, cmap='gray')
        plt.title(f'PC {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Project the first 100 images onto the first two principal components
    first_100_images = images[:100]
    projected_100_images = np.dot(first_100_images - np.mean(first_100_images, axis=0), pca_components[:,:2])

    # Labels for the first 100 images
    first_100_labels = labels[:100]

    # Visualization
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.unique(first_100_labels))
    for i, color in zip(range(10), colors):
        idx = first_100_labels == i
        plt.scatter(projected_100_images[idx, 0], projected_100_images[idx, 1], c=[color], label=i)
    plt.title("Projection of MNIST digits onto the first two principal components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Digit Label")
    plt.grid(True)
    plt.show()

    # Apply PCA2
    mean_images, pca_components2 = pca(images, 784, 3)

    # The first image in the dataset
    first_image = images[0]

    # List of k values
    k_values = [1, 50, 100, 250, 500, 784]

    # Reconstruct the first image for each k
    reconstructed_images = [reconstruct_image(first_image, pca_components2, mean_images, k) for k in k_values]

    # Visualization
    plt.figure(figsize=(15, 10))
    for i, image in enumerate(reconstructed_images):
        plt.subplot(2, 3, i + 1)
        plt.imshow(image.reshape(28, 28), cmap='Greys_r')
        plt.title(f'k = {k_values[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()