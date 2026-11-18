# Logistic Regression Cat Classifier

A from-scratch implementation of logistic regression for binary image classification (cat vs. non-cat), built with NumPy only (no external machine learning libraries (like scikit-learn) are used for modeling).

## Purpose

Logistic regression is the simplest building block of neural networks, yet it is often used as a black box through libraries like scikit-learn. This project implements it explicitly, covering forward propagation, the binary cross-entropy loss, backpropagation, and batch gradient descent, so that each step of the learning process is visible and testable.

The model is applied to a practical computer vision task: deciding whether an image contains a cat.

## What It Does

- Loads a labeled dataset of color images stored in HDF5 format.
- Flattens and normalizes the images into feature vectors.
- Trains a logistic regression classifier using batch gradient descent.
- Sigmoid activation and binary cross-entropy loss.
- Reports training and test accuracy.
- Includes unit tests that verify the cost, gradients, parameter updates, and predictions against known values.

## Who Can Benefit

- **Students and self-learners** who want to understand how logistic regression and gradient descent work beneath the surface.
- **Instructors** looking for a compact, readable reference implementation for teaching the fundamentals of neural networks.
- **Practitioners** who want a minimal baseline for image classification, or a refresher on the mathematics behind more complex models.

## Project Structure

| File | Description |
|------|-------------|
| `logistic_regression.py` | Model implementation: parameter initialization, propagation, training, prediction, and evaluation |
| `data_utils.py` | Dataset loading and label reshaping |
| `test_logistic_regression.py` | Unit tests for the core model functions |
| `datasets/` | Training and test data in `.h5` format |

## Data Description

The dataset contains color images in `.h5` format:

- **Training set:**  
  - `train_set_x`: shape (m_train, num_px, num_px, 3)  
  - `train_set_y`: shape (1, m_train)
- **Test set:**  
  - `test_set_x`: shape (m_test, num_px, num_px, 3)  
  - `test_set_y`: shape (1, m_test)
- **Labels:**  
  - 1 = Cat  
  - 0 = Non-cat  
- **Classes:**  
  - `list_classes`: [b'non-cat', b'cat']

Images are flattened and normalized before being fed to the model.

## Acknowledgements

The model design and dataset are inspired by the Deep Learning Specialization by Andrew Ng on Coursera.
