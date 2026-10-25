# importing the dataset from scikit-learn
from sklearn.datasets import load_digits

# loading the handwritten digits dataset
digits = load_digits()

# separating images (features) and lables (targets)
X = digits.images
Y = digits.target

# exploring basic information about the dataset
print("Dataset loaded successfully!")
print("Number of images:", len(digits.images))
print("Image shape:", digits.images.shape)
print("Features shape:", X.shape)
print("Labels shape:", Y.shape)
print("Available digits:",digits.target_names)