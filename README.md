# Dogs vs Cats Image Classification

## Overview

This project implements a Dogs vs Cats Image Classification system using Traditional Machine Learning techniques. Images are preprocessed using OpenCV and classified using Logistic Regression.

## Features

* Image preprocessing using OpenCV
* Image resizing and feature extraction
* Train-Test Split using Scikit-Learn
* Logistic Regression classifier
* Model evaluation using Accuracy Score and Confusion Matrix
* Model saving using Pickle

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-Learn
* Pickle

## Workflow

1. Load image dataset
2. Resize images
3. Flatten image pixels into feature vectors
4. Split dataset into training and testing sets
5. Train Logistic Regression model
6. Evaluate model performance
7. Save trained model

## Project Structure

Dogs-vs-Cats-Image-Classification/

├── dataset/

│ ├── cats/

│ └── dogs/

├── train.py

├── predict.py

├── model.pkl

└── README.md

## Future Improvements

* Support larger datasets
* Implement feature engineering
* Upgrade to Convolutional Neural Networks (CNN)
* Real-time image prediction
