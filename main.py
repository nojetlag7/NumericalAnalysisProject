"""
Numerical Methods Stock Price Prediction Project

This project compares three interpolation methods for generating synthetic stock data:
1. Lagrange interpolation
2. Newton's divided difference interpolation  
3. Newton's forward difference (for extrapolation)

Each method generates 50 synthetic points from the first 50 real values,
then trains a LinearRegression model to predict the remaining 150 prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')


class InterpolationMethods:
    """Class containing different interpolation methods for synthetic data generation"""
    
    @staticmethod
    def lagrange_interpolation(x_points, y_points, x_target):
        """
        Lagrange interpolation method (with numerical stability improvements)
        
        Args:
            x_points: Known x values
            y_points: Known y values  
            x_target: Target x values to interpolate
            
        Returns:
            Interpolated y values
        """
        n = len(x_points)
        result = np.zeros_like(x_target, dtype=np.float64)
        
        # Limit the number of points to prevent overflow (use every 3rd point if > 15)
        if n > 15:
            indices = np.arange(0, n, 3)
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
        
        for i in range(len(x_target)):
            total = 0.0
            for j in range(n):
                # Calculate Lagrange basis polynomial with overflow protection
                term = y_points[j]
                for k in range(n):
                    if k != j:
                        denominator = x_points[j] - x_points[k]
                        if abs(denominator) < 1e-10:  # Avoid division by very small numbers
                            continue
                        term *= (x_target[i] - x_points[k]) / denominator
                        
                        # Check for overflow
                        if abs(term) > 1e10:
                            term = np.sign(term) * 1e10
                            
                total += term
            result[i] = total
            
        return result
    
    @staticmethod
    def newton_divided_difference(x_points, y_points, x_target):
        """
        Newton's divided difference interpolation method (with stability improvements)
        
        Args:
            x_points: Known x values
            y_points: Known y values
            x_target: Target x values to interpolate
            
        Returns:
            Interpolated y values
        """
        n = len(x_points)
        
        # Limit the number of points to prevent overflow
        if n > 15:
            indices = np.arange(0, n, 3)
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
        
        # Create divided difference table
        divided_diff = np.zeros((n, n), dtype=np.float64)
        divided_diff[:, 0] = y_points
        
        for j in range(1, n):
            for i in range(n - j):
                denominator = x_points[i + j] - x_points[i]
                if abs(denominator) < 1e-10:  # Avoid division by very small numbers
                    divided_diff[i, j] = 0
                else:
                    divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / denominator
        
        # Evaluate Newton polynomial at target points
        result = np.zeros_like(x_target, dtype=np.float64)
        for i, x in enumerate(x_target):
            value = divided_diff[0, 0]
            for j in range(1, n):
                term = divided_diff[0, j]
                for k in range(j):
                    term *= (x - x_points[k])
                    # Check for overflow
                    if abs(term) > 1e10:
                        term = np.sign(term) * 1e10
                        break
                value += term
            result[i] = value
            
        return result
    
    @staticmethod
    def newton_forward_difference(x_points, y_points, x_target):
        """
        Newton's forward difference method (for extrapolation) with stability improvements
        
        Args:
            x_points: Known x values (assumed equally spaced)
            y_points: Known y values
            x_target: Target x values to extrapolate
            
        Returns:
            Extrapolated y values
        """
        n = len(x_points)
        h = x_points[1] - x_points[0]  # Step size
        
        # Limit the number of points to prevent overflow
        if n > 15:
            indices = np.arange(0, n, 3)
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
            h = x_points[1] - x_points[0]  # Recalculate step size
        
        # Create forward difference table
        forward_diff = np.zeros((n, n), dtype=np.float64)
        forward_diff[:, 0] = y_points
        
        for j in range(1, n):
            for i in range(n - j):
                forward_diff[i, j] = forward_diff[i + 1, j - 1] - forward_diff[i, j - 1]
        
        # Evaluate Newton forward polynomial at target points
        result = np.zeros_like(x_target, dtype=np.float64)
        for i, x in enumerate(x_target):
            # Calculate u = (x - x0) / h
            u = (x - x_points[0]) / h
            
            value = forward_diff[0, 0]
            u_term = 1.0
            for j in range(1, min(n, 15)):  # Limit terms to prevent overflow
                u_term *= (u - (j - 1))
                factorial_j = math.factorial(j)
                term = (u_term * forward_diff[0, j]) / factorial_j
                
                # Check for overflow or very large terms
                if abs(term) > 1e10 or np.isinf(term) or np.isnan(term):
                    break
                    
                value += term
            
            result[i] = value
            
        return result


class StockPricePredictor:
    """Main class for stock price prediction using interpolation methods"""
    
    def __init__(self, ticker='AAPL', days=200):
        """
        Initialize the predictor
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of historical data to fetch
        """
        self.ticker = ticker
        self.days = days
        self.data = None
        self.prices = None
        self.results = {}
        
    def fetch_stock_data(self):
        """Use hardcoded stock-like price data"""
        print(f"Using hardcoded stock-like data for {self.ticker}")
        
        # Hardcoded realistic stock price data (200 values)
        # This simulates a stock that starts around $150 and has realistic price movements
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
        
        # Use the exact number of days requested
        self.prices = np.array(hardcoded_prices[:self.days])
        
        # Create a simple date range for plotting
        dates = pd.date_range(start='2024-01-01', periods=len(self.prices), freq='D')
        self.data = pd.DataFrame({'Close': self.prices}, index=dates)
        
        print(f"Loaded {len(self.prices)} hardcoded stock prices")
        return True
    
    def generate_synthetic_data(self, method_name, interpolation_func):
        """
        Generate synthetic data using specified interpolation method
        
        Args:
            method_name: Name of the interpolation method
            interpolation_func: Function to perform interpolation
            
        Returns:
            Combined dataset (50 real + 50 synthetic values)
        """
        # Use first 50 real values
        x_real = np.arange(50)
        y_real = self.prices[:50]
        
        # Generate 50 synthetic points for days 50-99
        x_synthetic = np.arange(50, 100)
        
        try:
            y_synthetic = interpolation_func(x_real, y_real, x_synthetic)
            
            # Apply bounds to prevent extreme values
            y_min, y_max = np.min(y_real), np.max(y_real)
            price_range = y_max - y_min
            
            # Clamp synthetic values to reasonable bounds
            lower_bound = y_min - 0.5 * price_range
            upper_bound = y_max + 0.5 * price_range
            
            y_synthetic = np.clip(y_synthetic, lower_bound, upper_bound)
            
            print(f"  Generated synthetic data for {method_name}")
            print(f"  Original range: [{y_min:.2f}, {y_max:.2f}]")
            print(f"  Synthetic range: [{np.min(y_synthetic):.2f}, {np.max(y_synthetic):.2f}]")
            
        except Exception as e:
            print(f"  Warning: {method_name} failed, using linear interpolation fallback")
            # Fallback to simple linear interpolation
            y_synthetic = np.interp(x_synthetic, x_real, y_real)
        
        # Combine real and synthetic data
        x_combined = np.arange(100)
        y_combined = np.concatenate([y_real, y_synthetic])
        
        return x_combined, y_combined
    
    def train_and_predict(self, method_name, x_train, y_train):
        """
        Train LinearRegression model and predict remaining prices
        
        Args:
            method_name: Name of the interpolation method
            x_train: Training features (days 0-99)
            y_train: Training targets (prices for days 0-99)
            
        Returns:
            Predictions for remaining days
        """
        # Prepare training data
        X_train = x_train.reshape(-1, 1)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict remaining prices (from day 100 to end of available data)
        remaining_days = len(self.prices) - 100
        if remaining_days <= 0:
            print(f"Warning: Not enough data for prediction phase in {method_name}")
            return np.array([])
        
        x_test = np.arange(100, 100 + remaining_days)
        X_test = x_test.reshape(-1, 1)
        y_pred = model.predict(X_test)
        
        # Calculate metrics against actual prices
        y_actual = self.prices[100:100 + remaining_days]
        
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        # Store results
        self.results[method_name] = {
            'model': model,
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_pred': y_pred,
            'y_actual': y_actual,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        return y_pred
    
    def run_comparison(self):
        """Run the complete comparison of all three methods"""
        print("Starting numerical methods comparison...")
        
        # Fetch data
        if not self.fetch_stock_data():
            return
        
        # Initialize interpolation methods
        methods = InterpolationMethods()
        
        # Method 1: Lagrange interpolation
        print("\n1. Running Lagrange interpolation...")
        x_train, y_train = self.generate_synthetic_data(
            "Lagrange", methods.lagrange_interpolation
        )
        self.train_and_predict("Lagrange", x_train, y_train)
        
        # Method 2: Newton's divided difference
        print("2. Running Newton's divided difference...")
        x_train, y_train = self.generate_synthetic_data(
            "Newton Divided Diff", methods.newton_divided_difference
        )
        self.train_and_predict("Newton Divided Diff", x_train, y_train)
        
        # Method 3: Newton's forward difference
        print("3. Running Newton's forward difference...")
        x_train, y_train = self.generate_synthetic_data(
            "Newton Forward Diff", methods.newton_forward_difference
        )
        self.train_and_predict("Newton Forward Diff", x_train, y_train)
        
        print("All methods completed!")
    
    def display_results(self):
        """Display comparison results in a formatted table"""
        if not self.results:
            print("No results to display. Run comparison first.")
            return
        
        print("\n" + "="*80)
        print("NUMERICAL METHODS COMPARISON RESULTS")
        print("="*80)
        
        # Create results table
        print(f"{'Method':<25} {'MSE':<15} {'MAE':<15} {'R² Score':<15}")
        print("-" * 70)
        
        for method, results in self.results.items():
            print(f"{method:<25} {results['mse']:<15.4f} {results['mae']:<15.4f} {results['r2']:<15.4f}")
        
        # Find best method
        best_method = min(self.results.keys(), key=lambda x: self.results[x]['mse'])
        print(f"\nBest performing method (lowest MSE): {best_method}")
        
        return self.results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if not self.results:
            print("No results to visualize. Run comparison first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Stock Price Prediction Comparison - {self.ticker}', fontsize=16)
        
        # Plot 1: All predictions vs actual
        ax1 = axes[0, 0]
        x_actual = np.arange(len(self.prices))
        ax1.plot(x_actual, self.prices, 'k-', linewidth=2, label='Actual Prices', alpha=0.7)
        
        colors = ['red', 'blue', 'green']
        for i, (method, results) in enumerate(self.results.items()):
            # Plot training data (0-99)
            ax1.plot(results['x_train'], results['y_train'], 
                    color=colors[i], linestyle='--', alpha=0.6, 
                    label=f'{method} Training')
            
            # Plot predictions (100-199)
            ax1.plot(results['x_test'], results['y_pred'], 
                    color=colors[i], linewidth=2, 
                    label=f'{method} Prediction')
        
        ax1.axvline(x=50, color='orange', linestyle=':', alpha=0.7, label='Real/Synthetic Split')
        ax1.axvline(x=100, color='purple', linestyle=':', alpha=0.7, label='Train/Test Split')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price Predictions Comparison')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Metrics comparison
        ax2 = axes[0, 1]
        methods = list(self.results.keys())
        mse_values = [self.results[m]['mse'] for m in methods]
        
        bars = ax2.bar(methods, mse_values, color=['red', 'blue', 'green'], alpha=0.7)
        ax2.set_ylabel('Mean Squared Error')
        ax2.set_title('MSE Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, mse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 3: R² Score comparison
        ax3 = axes[1, 0]
        r2_values = [self.results[m]['r2'] for m in methods]
        
        bars = ax3.bar(methods, r2_values, color=['red', 'blue', 'green'], alpha=0.7)
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, r2_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Residuals for best method
        ax4 = axes[1, 1]
        best_method = min(self.results.keys(), key=lambda x: self.results[x]['mse'])
        best_results = self.results[best_method]
        
        residuals = best_results['y_actual'] - best_results['y_pred']
        ax4.scatter(best_results['x_test'], residuals, alpha=0.6, color='red')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Residuals')
        ax4.set_title(f'Residuals - {best_method} (Best Method)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Additional detailed plot
        self.create_detailed_plot()
    
    def create_detailed_plot(self):
        """Create a detailed plot showing the interpolation process"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Detailed Interpolation Analysis - {self.ticker}', fontsize=16)
        
        colors = ['red', 'blue', 'green']
        method_names = list(self.results.keys())
        
        for i, (method, results) in enumerate(self.results.items()):
            ax = axes[i]
            
            # Plot actual data
            x_actual = np.arange(len(self.prices))
            ax.plot(x_actual, self.prices, 'k-', linewidth=2, label='Actual Prices', alpha=0.7)
            
            # Highlight first 50 real values used for interpolation
            ax.scatter(np.arange(50), self.prices[:50], color='orange', s=30, 
                      label='Original 50 Points', zorder=5)
            
            # Plot synthetic data (50-99)
            synthetic_x = np.arange(50, 100)
            synthetic_y = results['y_train'][50:]
            ax.plot(synthetic_x, synthetic_y, color=colors[i], linestyle='--', 
                   linewidth=2, label=f'{method} Synthetic', alpha=0.8)
            
            # Plot predictions (100-199)
            ax.plot(results['x_test'], results['y_pred'], color=colors[i], 
                   linewidth=2, label=f'{method} Prediction')
            
            # Add vertical lines for phases
            ax.axvline(x=50, color='orange', linestyle=':', alpha=0.7, label='Interpolation Start')
            ax.axvline(x=100, color='purple', linestyle=':', alpha=0.7, label='Prediction Start')
            
            # Add metrics text
            ax.text(0.02, 0.98, f'MSE: {results["mse"]:.4f}\nMAE: {results["mae"]:.4f}\nR²: {results["r2"]:.4f}',
                   transform=ax.transAxes, verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Days')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{method} Method')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the stock price prediction comparison"""
    print("Numerical Methods Stock Price Prediction")
    print("="*50)
    
    # Initialize predictor
    predictor = StockPricePredictor(ticker='AAPL', days=200)
    
    # Run comparison
    predictor.run_comparison()
    
    # Display results
    results = predictor.display_results()
    
    # Create visualizations
    predictor.create_visualizations()
    
    # Additional analysis
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    if results:
        best_method = min(results.keys(), key=lambda x: results[x]['mse'])
        worst_method = max(results.keys(), key=lambda x: results[x]['mse'])
        
        print(f"Best performing method: {best_method}")
        print(f"Worst performing method: {worst_method}")
        
        best_mse = results[best_method]['mse']
        worst_mse = results[worst_method]['mse']
        improvement = ((worst_mse - best_mse) / worst_mse) * 100
        
        print(f"Performance improvement: {improvement:.2f}%")
        
        # Method characteristics
        print("\nMethod Characteristics:")
        print("- Lagrange: Global polynomial, can have oscillations")
        print("- Newton Divided Diff: Efficient for adding new points")
        print("- Newton Forward Diff: Better for extrapolation with equal spacing")


if __name__ == "__main__":
    main()
