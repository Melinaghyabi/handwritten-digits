# Step1: importing needed libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# Step2: loading the dataset
digits = load_digits()

# Step3: separating the features and target labels
X = digits.data #using data instead of images in order to flatten the image and pixels in a row to work easier in machine learning
Y = digits.target

# Step4: data structure
print("Number of samples:", X.shape[0]) # expected output: Number of samples: 1797
print("Number of features:", X.shape[1]) # expeected output: Number of features: 64
print("Shape of images:", digits.images.shape) # expected output: Shape of images: (1797, 8, 8)
print("Target labels:", digits.target_names) # expected output: Target labels: [0 1 2 3 4 5 6 7 8 9]
print("Task type: Classification") # expected output: Task type: Classification

# Step4: Basic statistics
# in this dataset it tells us how bright the pixels are across the dataset
print("Maximum pixel value:", np.max(X)) # expected output: Maximum pixel value: 16
print("Minimum pixel value:", np.min(X)) # expected output: Minimun pixel value: 0
print("Average pixel value:", np.mean(X)) # tells us the average brightness of all the pixels across the dataset
print("Standard deviation:", np.std(X)) # tells us how much these pixel brightness values vary around the dataset

# Step5: Class distribution
# we count how many exxamples of each digits exist
labels, counts = np.unique(Y, return_counts = True)

for label, counts in zip(labels, counts):
    print(f"Digit {label}: {counts} images") # example for expected oputput: Digit 0: 178 images, it print for all the digits from 0 to 9

# Step6: Missing value check
missing_values = np.isnan(X).sum()

print("Missing values:", missing_values) 
# if a pixel's brightness were unknown, it could be represented as NaN

# Step7: Display a sample image
plt.imshow(digits.images[0], cmap="gray")
plt.title(f"Handwritten Digits: {Y[0]}")
plt.axis("off")
plt.show()
# this shows the first image and its correct digit label, 0