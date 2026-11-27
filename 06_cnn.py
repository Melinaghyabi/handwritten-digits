# Step 1: importing the tools we need
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os
os.makedirs("figures", exist_ok = True)
keras.utils.set_random_seed(42)  # makes the CNN results repeatable

# Step 2: loading the dataset
digits = load_digits()
X = digits.data    # pixel values for each image
Y = digits.target  # correct digit for each image

# Step 3: splitting the data (same split as 04_classify_digits.py and 05_svm.py)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size = 0.2,
    random_state = 42,
    stratify = Y
)

# Step 4: reshaping for the CNN (64 numbers -> 8x8 image with 1 channel), scaled to 0-1
X_train_cnn = X_train.reshape(-1, 8, 8, 1) / 16.0
X_test_cnn = X_test.reshape(-1, 8, 8, 1) / 16.0

print("Training shape:", X_train_cnn.shape)
print("Testing shape:", X_test_cnn.shape)

# Step 5: building the CNN, layer by layer
model = keras.Sequential([
    layers.Input(shape = (8, 8, 1)),                                     # one 8x8 grayscale image
    layers.Conv2D(32, (3, 3), activation = "relu", padding = "same"),    # 32 pattern detectors
    layers.MaxPooling2D((2, 2)),                                         # shrink 8x8 down to 4x4
    layers.Conv2D(64, (3, 3), activation = "relu", padding = "same"),    # 64 detectors for bigger shapes
    layers.Flatten(),                                                    # turn the grid into one long list
    layers.Dense(64, activation = "relu"),                               # combine the patterns
    layers.Dense(10, activation = "softmax")                             # 10 outputs: one per digit
])

model.summary()

# Step 6: choosing how the CNN learns
model.compile(
    optimizer = "adam",                           # the method for adjusting the parameters
    loss = "sparse_categorical_crossentropy",     # measures how wrong the guesses are
    metrics = ["accuracy"]
)

# Step 7: training
history = model.fit(
    X_train_cnn, Y_train,
    epochs = 30,              # go through all training images 30 times
    batch_size = 32,          # update the parameters after every 32 images
    validation_split = 0.1,   # hold back 10% of the training images to check progress
    verbose = 1
)

# Step 8: the final exam on the 360 test images
test_loss, test_accuracy = model.evaluate(X_test_cnn, Y_test, verbose = 0)
Y_pred_cnn = model.predict(X_test_cnn, verbose = 0).argmax(axis = 1)  # pick the digit with the highest score

print(f"CNN test accuracy: {test_accuracy:.2%}")
print("Number of incorrect predictions:", (Y_test != Y_pred_cnn).sum())

# Step 9: training curve - accuracy for every epoch
plt.figure(figsize = (6, 4))
plt.plot(history.history["accuracy"], label = "Training accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
plt.title("CNN accuracy during training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("figures/cnn_training_curve.png")
plt.show()

# Step 10: CNN confusion matrix
# rows = the true digit, columns = what the model predicted
ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_cnn, cmap = "Blues")
plt.title("CNN confusion matrix")
plt.savefig("figures/cnn_confusion_matrix.png")
plt.show()

# Step 11: the images the CNN got wrong
wrong = (Y_test != Y_pred_cnn).nonzero()[0]  # positions of the wrong predictions

plt.figure(figsize = (2 * len(wrong), 2.5))
for i, index in enumerate(wrong):
    plt.subplot(1, len(wrong), i + 1)                       # one small box per wrong image
    plt.imshow(X_test[index].reshape(8, 8), cmap = "gray")  # turn 64 numbers back into an 8x8 picture
    plt.title(f"True: {Y_test[index]}\nPred: {Y_pred_cnn[index]}")
    plt.axis("off")
plt.savefig("figures/cnn_misclassified.png")
plt.show()

# Step 12: re-running KNN and SVM on the same split (flat 64-number rows, like before)
knn = KNeighborsClassifier(n_neighbors = 3).fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)

svm = SVC(C = 10, gamma = 0.001).fit(X_train, Y_train)  # the best settings found in 05_svm.py
Y_pred_svm = svm.predict(X_test)

# Step 13: one results table for all three models
predictions = {
    "KNN (k=3)": Y_pred_knn,
    "SVM": Y_pred_svm,
    "CNN": Y_pred_cnn
}

for name, pred in predictions.items():
    accuracy = accuracy_score(Y_test, pred)
    f1 = f1_score(Y_test, pred, average = "macro")
    mistakes = (Y_test != pred).sum()
    print(f"{name:10s} accuracy: {accuracy:.2%}   macro F1: {f1:.3f}   mistakes: {mistakes}")

# Step 14: bar chart - number of mistakes per model
names = list(predictions.keys())
mistakes = [(Y_test != pred).sum() for pred in predictions.values()]
accuracies = [accuracy_score(Y_test, pred) for pred in predictions.values()]

plt.figure(figsize = (6, 4))
bars = plt.bar(names, mistakes, color = ["#9bb7d4", "#2b5c8a", "#c97b63"], edgecolor = "black")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             f"{acc:.2%}", ha = "center")                        # accuracy written above each bar
plt.title("Mistakes on the 360 test images (fewer is better)")
plt.ylabel("Number of incorrect predictions")
plt.savefig("figures/model_comparison.png", bbox_inches = "tight")
plt.show()
