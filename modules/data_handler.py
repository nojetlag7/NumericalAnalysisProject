import numpy as np
import pandas as pd

class DataHandler:
    # This class handles loading stock data for gap-filling analysis
    # Provides 200 hardcoded stock prices that simulate realistic market movements
    def __init__(self, ticker='AAPL', days=200):
        self.ticker = ticker
        self.days = days
        self.data = None
        self.prices = None

    def fetch_stock_data(self):
        """
        # Load hardcoded stock price data for gap-filling analysis
        # For this project, we use 200 fixed prices that simulate realistic stock movements
        # This ensures consistent results and avoids external API dependencies
        """
        print(f"Using hardcoded stock-like data for {self.ticker}")
        hardcoded_prices = [
            150.25, 151.30, 149.80, 152.15, 153.40, 151.90, 154.20, 155.60, 152.80, 156.30,
            157.75, 155.40, 158.90, 160.25, 157.60, 161.80, 163.20, 160.45, 164.70, 166.15,
            163.30, 167.50, 169.80, 166.20, 170.40, 172.85, 169.60, 173.20, 175.40, 172.10,
            176.80, 178.25, 175.90, 179.60, 181.30, 178.45, 182.70, 184.15, 181.80, 185.40,
            187.60, 184.25, 188.90, 190.35, 187.50, 191.80, 193.25, 190.90, 194.60, 196.15,
            193.40, 197.80, 199.25, 196.60, 200.40, 202.85, 199.20, 203.60, 205.15, 202.80,
            206.40, 208.25, 205.90, 209.60, 211.30, 208.45, 212.70, 214.15, 211.80, 215.40,
            217.60, 214.25, 218.90, 220.35, 217.50, 221.80, 223.25, 220.90, 224.60, 226.15,
            223.40, 227.80, 229.25, 226.60, 230.40, 232.85, 229.20, 233.60, 235.15, 232.80,
            236.40, 238.25, 235.90, 239.60, 241.30, 238.45, 242.70, 244.15, 241.80, 245.40,
            247.60, 244.25, 248.90, 250.35, 247.50, 251.80, 253.25, 250.90, 254.60, 256.15,
            253.40, 257.80, 259.25, 256.60, 260.40, 262.85, 259.20, 263.60, 265.15, 262.80,
            266.40, 268.25, 265.90, 269.60, 271.30, 268.45, 272.70, 274.15, 271.80, 275.40,
            277.60, 274.25, 278.90, 280.35, 277.50, 281.80, 283.25, 280.90, 284.60, 286.15,
            283.40, 287.80, 289.25, 286.60, 290.40, 292.85, 289.20, 293.60, 295.15, 292.80,
            296.40, 298.25, 295.90, 299.60, 301.30, 298.45, 302.70, 304.15, 301.80, 305.40,
            307.60, 304.25, 308.90, 310.35, 307.50, 311.80, 313.25, 310.90, 314.60, 316.15,
            313.40, 317.80, 319.25, 316.60, 320.40, 322.85, 319.20, 323.60, 325.15, 322.80,
            326.40, 328.25, 325.90, 329.60, 331.30, 328.45, 332.70, 334.15, 331.80, 335.40,
            337.60, 334.25, 338.90, 340.35, 337.50, 341.80, 343.25, 340.90, 344.60, 346.15
        ]
        
        # Ensure we have enough data
        if len(hardcoded_prices) < self.days:
            print(f"Warning: Only {len(hardcoded_prices)} prices available, but {self.days} requested")
            self.days = len(hardcoded_prices)
        
        self.prices = np.array(hardcoded_prices[:self.days])
        dates = pd.date_range(start='2024-01-01', periods=len(self.prices), freq='D')
        self.data = pd.DataFrame({'Close': self.prices}, index=dates)
        print(f"Loaded {len(self.prices)} hardcoded stock prices")
        return True

    def get_prices(self):
        return self.prices

    def get_data(self):
        return self.data
