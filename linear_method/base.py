import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
# Python

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from utilis.evaluate import plot_predictions


class BaseModel:
    model = None
    mse = 0.0

    def __init__(self, x, y, n_components=2, cv_folds=None):
        self.x = x
        self.y = y
        self.n_components = n_components
        self.cv_fold = cv_folds

    def fit(self):
        self.model = PLSRegression(n_components=self.n_components)

        self.y_pred = cross_val_predict(self.model, self.x, self.y, cv=self.cv_fold)

        self.mse = mean_squared_error(self.y, self.y_pred)

        self.model.fit(self.x, self.y)

    def evaluate(self):
        plot_predictions(self.y_pred, self.y)


def fit_spectral_data(X, y, n_components=2, cv_folds=5):
    """
    Fit spectral data using Partial Least Squares (PLS) regression with cross-validation.

    Args:
    - X: ndarray, shape (n_samples, n_features), input spectral data
    - y: ndarray, shape (n_samples,), target values
    - n_components: int, number of components to keep in the PLS regression (default is 2)
    - cv_folds: int, number of folds for cross-validation (default is 5)

    Returns:
    - pls_model: PLSRegression object, the fitted PLS regression model
    - y_pred: ndarray, shape (n_samples,), predicted target values using cross-validation
    - mse_cv: float, mean squared error of cross-validated predictions
    """

    # Initialize PLS regression model
    pls_model = PLSRegression(n_components=n_components)

    # Perform cross-validation
    y_pred = cross_val_predict(pls_model, X, y, cv=cv_folds)

    # Compute mean squared error of cross-validated predictions
    mse_cv = mean_squared_error(y, y_pred)

    # Fit the PLS regression model on the entire dataset
    pls_model.fit(X, y)

    return pls_model, y_pred, mse_cv


if __name__ == "__main__":
    n_com = 6
    cv_fold = 5

    df = pd.read_excel("../data/Intra_CLS.xlsx")
    x = df.iloc[:, 3:]
    y = df.iloc[:, 2]
    x = np.array(x)
    y = np.array(y)

    m = BaseModel(x, y, n_com, cv_fold)

    m.fit()
    m.evaluate()
