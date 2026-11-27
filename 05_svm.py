# step one: importing the tools we need 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# step two: loeading the dataset
digits= load_digits()
x = digits.data  #pixel value that contains the data
y = digits.target  # correct digits for the image

#step three: splitting the data set to fit the KNN 
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y,
 test_size = 0.2,
 random_state = 42,
 stratify = y   
)

# Step four: creating the SVM model
model = SVC()  # using the default settings first

# Step five: training the model on the training images
model.fit(X_train, Y_train)

# Step six: predicting the digits in the 360 testing images
Y_pred = model.predict(X_test)

# Step seven: measuring accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"SVM accuracy: {accuracy:.2%}")

# counting the mistakes
incorrect = (Y_test != Y_pred).sum()
print("Number of incorrect predictions:", incorrect)

# Step eight: tuning the SVM settings with cross-validation on the training data
settings = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.001, 0.01]
}

search = GridSearchCV(SVC(), settings, cv = 5)  # tries all 12 combinations, 5 folds each
search.fit(X_train, Y_train)

print("Best settings:", search.best_params_)
print(f"Best cross-validation accuracy: {search.best_score_:.2%}")

# testing the best settings on the unseen test images
best_model = search.best_estimator_
Y_pred_best = best_model.predict(X_test)
print(f"Tuned SVM test accuracy: {accuracy_score(Y_test, Y_pred_best):.2%}")
print("Number of incorrect predictions:", (Y_test != Y_pred_best).sum())

# Step nine: confusion matrix for the tuned SVM
# rows = the true digit, columns = what the model predicted
ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred_best, cmap = "Blues")
plt.title("SVM confusion matrix")
plt.savefig("figures/svm_confusion_matrix.png")
plt.close()  # save the chart without opening a window

# Step ten: showing the images the SVM got wrong
wrong = (Y_test != Y_pred_best).nonzero()[0]  # positions of the wrong predictions

plt.figure(figsize = (6, 2.5))
for i, index in enumerate(wrong):
    plt.subplot(1, len(wrong), i + 1)                       # one small box per wrong image
    plt.imshow(X_test[index].reshape(8, 8), cmap = "gray")  # turn 64 numbers back into an 8x8 picture
    plt.title(f"True: {Y_test[index]}\nPred: {Y_pred_best[index]}")
    plt.axis("off")
plt.savefig("figures/svm_misclassified.png")
plt.close()  # save the chart without opening a window