import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from linear_method.base import BaseModel


class CGPCA(BaseModel):

    def __init__(self, x, y, n_components=2, cv_folds=None):
        super(CGPCA, self).__init__(x, y, n_components, cv_folds)

    def fit(self):
        self.model = PCA(self.n_components)

        self.y_pred = cross_val_predict(self.model, self.x, self.y, cv=self.cv_fold)

        self.mse = mean_squared_error(self.y, self.y_pred)

        self.model.fit(self.x, self.y)

class CGGPR(BaseModel):

    def __init__(self, x, y, kernel, n_components=2, cv_folds=None):
        super(CGGPR, self).__init__(x, y, n_components, cv_folds)
        self.kernel = kernel

    def fit(self):
        self.model = GaussianProcessRegressor(self.kernel)



if __name__ == "__main__":
    n_com = 6
    cv_fold = 5

    df = pd.read_excel("data/Intra_CLS.xlsx")
    x = df.iloc[:, 3:]
    y = df.iloc[:, 2]
    x = np.array(x)
    y = np.array(y)

    m = CGPCA(x, y, n_com, cv_fold)

    m.fit()
    m.evaluate()
