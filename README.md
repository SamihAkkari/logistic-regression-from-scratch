# Logistic Regression Cat Classifier

This repository implements a logistic regression model to perform binary image classification (cat vs. non-cat). The model and data handling code is inspired by the exercises in the Deep Learning Specialization (Coursera, Andrew Ng).

## Project Structure

- `logistic_regression.py`: Implements a logistic regression model from scratch using numpy.
- `data_utils.py`: Loads, preprocesses, and returns the dataset.
- `test_logistic_regression.py`: Unit tests for the logistic regression model.
- `datasets/`: Contains the original `.h5` datasets (`train_catvnoncat.h5`, `test_catvnoncat.h5`) as provided in the Deep Learning Specialization course.

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

## Model Details

- Implemented from scratch using only numpy.
- Trains using batch gradient descent.
- Sigmoid activation and binary cross-entropy loss.
- No external machine learning libraries (like scikit-learn) are used for modeling.

## Data Origin

The dataset is taken from the Deep Learning Specialization (Coursera, Andrew Ng), "Logistic Regression as a Neural Network" assignment.
