
from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data
Y = digits.target

print("Dataset shape:", X.shape)
print("Labels shape:", Y.shape)
print("Classes:", digits.target_names)