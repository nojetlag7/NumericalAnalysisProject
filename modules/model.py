import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StockModel:
    # This class handles training and prediction using LinearRegression
    def __init__(self):
        self.model = LinearRegression()
        self.results = {}

    def train(self, x_train, y_train):
        X_train = x_train.reshape(-1, 1)
        self.model.fit(X_train, y_train)

    def predict(self, x_test):
        X_test = x_test.reshape(-1, 1)
        return self.model.predict(X_test)

    def evaluate(self, y_actual, y_pred):
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        return mse, mae, r2
