# importing the dataset from scikit-learn
from sklearn.datasets import load_digits

# loading the handwritten digits dataset
digits = load_digits()

# separating images (features) and lables (targets)
X = digits.data
Y = digits.target

# exploring basic information about the dataset
print("Dataset loaded successfully!") # expected output: Dataset loaded successfully!
print("Number of images:", len(digits.images)) # expected output: Number of images: 1797
print("Image shape:", digits.images.shape) # expected output: Image shape: (1797, 8, 8)
print("Features shape:", X.shape) # expected output: Features shape: (1797, 64)
print("Labels shape:", Y.shape) # expected output: Labels shape: (1797,)
print("Available digits:", digits.target_names) # expected output: Available digits: [0 1 2 3 4 5 6 7 8 9]