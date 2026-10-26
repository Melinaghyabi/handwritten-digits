# loading the libraries
# for this type of dataset we move on with k-Nearest Neighbors, it predicts the label of a new image by looking for similar images it has seen before
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()

X = digits.data # X contains pixel values (1797, 64)
Y = digits.target # Y contains correct labels (1797,)

# spliting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size = 0.2, # reservig 20% of mages for testing
    random_state = 42, # making the random split reproducible
    stratify = Y # keeping roughly the same proportion of each digit in both sets
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))

# creating a KNN classifier
model = KNeighborsClassifier(n_neighbors = 3) # the model checks 3 closest training images when predicting a digit

# training the model
model.fit(X_train, Y_train) # preparing the classifier using the training images and their correct labels. For KNN, this mainly means storing those examples so that they can be compared with new images

# Predicting labels for the testing images
Y_pred = model.predict(X_test) # the classifier predicts which digits appear in the 360 (1797 x 20%) testing images

# calculating prediction accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print(f"Model accuracy: {accuracy: .2%}")

# comparing predictions with actual labels
print("First 10 actual labels:", Y_test[:10])
print("First 10 predctions:", Y_pred[:10])

# counting incorrect predictions
incorrect = (Y_test != Y_pred).sum()
print("Number of incorrect predictions:", incorrect)