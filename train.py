import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------

X = []
y = []

IMAGE_SIZE = 50

# Cats
for filename in os.listdir("dataset/cats"):
    path = os.path.join("dataset/cats", filename)

    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.flatten()

    X.append(image)
    y.append(0)

# Dogs
for filename in os.listdir("dataset/dogs"):
    path = os.path.join("dataset/dogs", filename)

    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.flatten()

    X.append(image)
    y.append(1)

X = np.array(X)
y = np.array(y)

print("Dataset Loaded Successfully")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Images :", X_train.shape[0])
print("Testing Images  :", X_test.shape[0])

# -----------------------------
# Train Model
# -----------------------------

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

print("Training Completed!")

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))