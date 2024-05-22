import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, train_test_split
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

    # df = pd.read_excel("../data/Aug_Intra.xlsx")
    df = pd.read_excel("../data/Intra_CLS1.xlsx")
    df_extra = pd.read_excel("../data/Intra-extra.xlsx")

    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    df_extra = scaler.transform(df_extra)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    model = PLSRegression(n_components=n_com)
    model.fit(train_df[:, 3:], train_df[:, 2])

    y_test = test_df[:, 2]
    x_test = test_df[:, 3:]
    y_pred = model.predict(x_test)
    r2 = model.score(x_test, y_pred)
    mse = mean_squared_error(y_pred, y_test)
    print(f"R2: {r2}, mse: {mse}, rmse: {np.sqrt(mse)}")

    y_extra = df_extra[:, 2]
    x_extra = df_extra[:, 3:]
    y_pred2 = model.predict(x_extra)
    r2 = model.score(x_extra, y_pred2)
    mse = mean_squared_error(y_pred2, y_extra)
    print(f"R2: {r2}, mse: {mse}, rmse: {np.sqrt(mse)}")

    # Visualize predicted vs actual values with different colors
    plt.scatter(y_test, y_pred, c='blue', label='Actual vs Predicted')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', c='red', label='Perfect Prediction')
    plt.xlabel("Actual Diabetes Progression")
    plt.ylabel("Predicted Diabetes Progression")
    plt.title("PLS Regression: Predicted vs Actual Diabetes Progression")
    plt.legend()
    plt.show()
