import numpy as np
import pandas as pd

class DataHandler:
    # This class handles loading stock data and generating synthetic data
    def __init__(self, ticker='AAPL', days=100):
        self.ticker = ticker
        self.days = days
        self.data = None
        self.prices = None

    def fetch_stock_data(self):
        """
        # Get the stock price data
        # For this project, we use more realistic stock-like data with random fluctuations,
        # trends, and occasional volatile movements that mimic real stock behavior.
        """
        print(f"Using realistic stock-like data for {self.ticker}")
        # More realistic stock prices with various patterns: trends, volatility, corrections
        # Using 100 data points for better numerical stability
        hardcoded_prices = [
            150.25, 148.90, 152.30, 149.80, 155.20, 153.40, 147.60, 151.90, 154.80, 158.30,
            155.60, 152.80, 149.40, 153.70, 157.25, 160.80, 158.20, 155.90, 162.40, 159.70,
            165.30, 168.50, 164.20, 161.80, 167.90, 172.30, 169.60, 165.40, 171.80, 175.20,
            172.90, 176.40, 174.10, 178.80, 181.50, 177.20, 174.90, 179.60, 182.30, 185.90,
            183.40, 180.70, 186.20, 189.80, 187.50, 184.30, 188.90, 192.40, 195.80, 191.20,
            188.60, 194.30, 197.80, 201.50, 198.20, 195.90, 199.60, 203.40, 207.20, 204.80,
            201.50, 206.80, 210.40, 207.90, 204.60, 208.30, 212.80, 216.50, 213.20, 210.90,
            215.60, 219.40, 216.80, 213.50, 218.20, 221.90, 225.60, 222.30, 219.80, 224.50,
            228.20, 225.90, 222.60, 227.30, 231.80, 228.40, 224.70, 229.60, 233.90, 237.20,
            234.80, 231.50, 236.20, 240.60, 237.90, 234.40, 239.80, 243.50, 246.20, 242.90
        ]
        self.prices = np.array(hardcoded_prices[:self.days])
        dates = pd.date_range(start='2024-01-01', periods=len(self.prices), freq='D')
        self.data = pd.DataFrame({'Close': self.prices}, index=dates)
        print(f"Loaded {len(self.prices)} realistic stock prices")
        return True

    def get_prices(self):
        return self.prices

    def get_data(self):
        return self.data
