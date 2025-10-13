import numpy as np
from logistic_regression import LogisticRegression 


def test_propagate_basic():
    """
    Unit test for the propagate function.
    Checks that gradients and cost are correct for a known input.
    """
    # Arrange
    model = LogisticRegression(learning_rate=0.001, num_iterations=2000)
    model.w = np.array([[1.], [2.]])
    model.b = 2.
    X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])

    # Act
    cost, gradients = model.propagate(X, Y, X.shape[1])

    expected_dw = np.array([[0.99845601], [2.39507239]])
    expected_db = 0.00145557813678
    expected_cost = 5.801545319394553

    assert np.allclose(gradients["dw"], expected_dw, atol=1e-6), "dw mismatch"
    assert np.isclose(gradients["db"], expected_db, atol=1e-6), "db mismatch"
    assert np.isclose(cost, expected_cost, atol=1e-6), "cost mismatch"

def test_fit_basic():
    """
    Unit test for the fit function
    Checks if the predictions are correct for a known example
    """
    model = LogisticRegression(learning_rate=0.001, num_iterations=2000)
    X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])
    model.fit(X, Y)

    w = model.w
    b = model.b

    expected_w = np.array([[-0.19999246], [-0.22045237]])
    expected_b = 0.39548853904436987

    assert np.allclose(w, expected_w, atol=1e-6), "w mismatch"
    assert np.isclose(b, expected_b, atol=1e-6), "b mismatch"

def test_prediction_basic():
    """
    Unit test for the prediction function
    Checks if the predictions are correct for a known example
    """
    model = LogisticRegression(learning_rate=0.001, num_iterations=2000)
    w = np.array([[0.1124579],[0.23106775]])
    b = -0.3
    model.w = w
    model.b = b
    X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
    preds = model.predict(X)
    expected_preds = np.array([[1., 1., 0.]])

    assert np.allclose(preds, expected_preds, atol=1e-6), "predictions mismatch"




