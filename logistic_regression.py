from data_utils import load_dataset
import numpy as np
from typing import Dict, Tuple


class LogisticRegression:
    
    def __init__(self, learning_rate:float = 0.001, num_iterations: int = 1000, verbose: bool = False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.w = None
        self.b = None
        self.costs = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-z))

    def initialize_parameters(self, n_features: int):
        """Initialize weights and bias to zeros."""
        self.w = np.zeros((n_features, 1))
        self.b = 0

    def propagate(self, X: np.ndarray, Y: np.ndarray, m: int):
        # FORWARD PROPAGATION
        Z = np.dot(self.w.T, X) + self.b
        A = self.sigmoid(Z)
        cost = -1/m * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        # BACKWARD PROPAGATION
        grad_w = 1/m * np.dot(X, (A - Y).T)
        grad_b = 1/m * np.sum(A - Y)

        gradients = {
            "dw" : grad_w,
            "db" : grad_b
        }
        return cost, gradients

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.initialize_parameters(X.shape[0])
        m = X.shape[1]
        for i in range(self.num_iterations):
            cost, gradients = self.propagate(X, Y, m)
            
            self.w = self.w - self.learning_rate * gradients["dw"]
            self.b = self.b - self.learning_rate * gradients["db"]

            if i % 100 == 0:
                self.costs.append(cost)
                if self.verbose:
                    print(f"Iteration {i}: cost = {cost:.6f}")
        if self.verbose:
            print("Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for a dataset."""
        Z = np.dot(self.w.T, X) + self.b
        A = self.sigmoid(Z)
        predictions = (A > 0.5).astype(int)
        return predictions

    def accuracy(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """Compute accuracy percentage."""
        return 100 * (1 - np.mean(np.abs(Y_pred - Y_true)))

if __name__ == "__main__":
    
    # PREPARE DATASET
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
    train_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T / 255.
    test_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T / 255.

    # Initialize and train model
    model = LogisticRegression(learning_rate=0.001, num_iterations=2000, verbose=True)
    model.fit(train_x_flatten, train_set_y)

    # Evaluate model
    train_preds = model.predict(train_x_flatten)
    test_preds = model.predict(test_x_flatten)

    print(f"Train accuracy: {model.accuracy(train_preds, train_set_y):.2f} %")
    print(f"Test accuracy:  {model.accuracy(test_preds, test_set_y):.2f} %")




