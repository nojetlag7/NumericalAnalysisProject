import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class StockModel:
    # This class handles training and prediction using LinearRegression
    # Used to predict missing stock prices based on known + interpolated data
    def __init__(self):
        self.model = LinearRegression()
        self.results = {}

    def train(self, x_train, y_train):
        # Train the linear regression model
        # x_train: Array of time indices (150 points: 100 known + 50 interpolated)
        # y_train: Array of stock prices (150 points: 100 known + 50 interpolated)
        X_train = x_train.reshape(-1, 1)
        self.model.fit(X_train, y_train)

    def predict(self, x_test):
        # Predict stock prices for given time indices
        # x_test: Array of time indices (50 points to predict)
        # Returns: Array of predicted stock prices (50 points)
        X_test = x_test.reshape(-1, 1)
        return self.model.predict(X_test)

    def evaluate(self, y_actual, y_pred):
        # Evaluate model performance using standard metrics
        # y_actual: True stock prices (50 points)
        # y_pred: Predicted stock prices (50 points)
        # Returns: MSE, MAE, R² score
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        return mse, mae, r2
