from sklearn.datasets import load_digits
digits = load_digits()
print(digits.images.shape)
print(digits.images[0])
print(digits.target[0])
print(digits.target[:10])
print(digits.images[:10])

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

for i in range (5):
    plt.imshow(digits.images[i], cmap="gray")
    plt.show()

fig, axes = plt.subplots(1, 2)

axes[0].imshow(digits.images[0], cmap="gray")
axes[1].imshow(digits.images[1], cmap="gray")

plt.show()