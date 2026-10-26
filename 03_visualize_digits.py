# importing libraries
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# loading dataset
digits = load_digits()
Y = digits.target

# Graph 1: Number of images for each digit
# counting examples of each digit
labels, counts = np.unique(Y, return_counts = True)

# creating bar chart
plt.bar(labels, counts,
        edgecolor = "black") # drawing a black outline around each bar

# adding titles and labels
plt.title("Number of images per digit")
plt.xlabel("Digits")
plt.ylabel("Number of images")
plt.xticks(labels) # making sure the horizontal axis shows every digit from 0 to 9 

# adding the graph to the figures folder
plt.savefig("figures/class_distribution.png")

# displaying the graph
plt.show()

# Graph 2: Distribution of pixel brightness
# getting all the pixel values in one long array
pixel_values = digits.data.flatten()

# creating histogram
plt.hist(pixel_values, bins = np.arange(18) - 0.5, # placing each integer pixel value in the center of its bin
        edgecolor = "black") # drawing a black outline around each bar

# adding titles and labels
plt.title("Distribution of pixel brightness")
plt.xlabel("Pixel intensity (0-16)")
plt.ylabel("Number of pixels")
plt.xticks(range(17))

#adding the graph to the figures folder
plt.savefig("figures/pixel_distribution.png")

# displaying the graph
plt.show()
