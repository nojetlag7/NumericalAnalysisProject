"""
Numerical Methods Stock Price Prediction Project

# This project shows how to use three different numerical methods to generate synthetic stock prices.
# Structure:
# - Dataset: 100 total data points for better numerical stability
# - Compute synthetic: Use first 50 real prices to create 25 synthetic prices (days 50-74)
# - Train model: Use first 75 points (50 real + 25 synthetic) to train LinearRegression
# - Analyze/predict: Use remaining 25 points (days 75-99) for testing and evaluation
# 
# The three numerical methods are:
# 1. Lagrange interpolation (traditional polynomial fitting approach)
# 2. Euler's method (differential equation approach for financial modeling)
# 3. Runge-Kutta 4th order (high-accuracy differential equation approach)
"""


# Import modules
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from modules.numerical_methods import NumericalMethods
from modules.data_handler import DataHandler
from modules.stock_model import StockModel
from modules.visualization import Visualization


def main():
    """
    # Main function to run the whole project
    # It sets up the modules, runs the comparison, shows results, and does extra analysis
    """
    print("Numerical Methods Stock Price Prediction")
    print("="*50)

    # Set up modules
    ticker = 'AAPL'
    days = 100  # Reduced dataset size for better numerical stability
    data_handler = DataHandler(ticker=ticker, days=days)
    numerical_methods = NumericalMethods()
    model = StockModel()
    viz = Visualization()

    # Fetch data
    if not data_handler.fetch_stock_data():
        return
    prices = data_handler.get_prices()

    results = {}
    # Method 1: Lagrange interpolation (traditional approach)
    print("\n1. Running Lagrange interpolation...")
    x_train, y_train = generate_synthetic_data(prices, "Lagrange", numerical_methods.lagrange_interpolation)
    y_pred, x_test, y_actual, mse, mae, r2 = train_and_predict(prices, x_train, y_train, model)
    results["Lagrange"] = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_pred': y_pred,
        'y_actual': y_actual,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    # Method 2: Euler's method (differential equation approach)
    print("2. Running Euler's method...")
    x_train, y_train = generate_synthetic_data(prices, "Euler Method", numerical_methods.euler_method)
    y_pred, x_test, y_actual, mse, mae, r2 = train_and_predict(prices, x_train, y_train, model)
    results["Euler Method"] = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_pred': y_pred,
        'y_actual': y_actual,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    # Method 3: Runge-Kutta 4th order (high-accuracy differential equation approach)
    print("3. Running Runge-Kutta 4th order...")
    x_train, y_train = generate_synthetic_data(prices, "Runge-Kutta 4th", numerical_methods.runge_kutta_4th_order)
    y_pred, x_test, y_actual, mse, mae, r2 = train_and_predict(prices, x_train, y_train, model)
    results["Runge-Kutta 4th"] = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_pred': y_pred,
        'y_actual': y_actual,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    print("All methods completed!")

    # Show results table
    print("\n" + "="*80)
    print("NUMERICAL METHODS COMPARISON RESULTS")
    print("="*80)
    print(f"{'Method':<25} {'MSE':<15} {'MAE':<15} {'R² Score':<15}")
    print("-" * 70)
    for method, res in results.items():
        print(f"{method:<25} {res['mse']:<15.4f} {res['mae']:<15.4f} {res['r2']:<15.4f}")
    best_method = min(results.keys(), key=lambda x: results[x]['mse'])
    print(f"\nBest performing method (lowest MSE): {best_method}")

    # Make the plots
    viz.plot_comparison(prices, results, ticker)
    viz.plot_detailed(prices, results, ticker)

    # Extra analysis and summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    worst_method = max(results.keys(), key=lambda x: results[x]['mse'])
    best_mse = results[best_method]['mse']
    worst_mse = results[worst_method]['mse']
    improvement = ((worst_mse - best_mse) / worst_mse) * 100
    print(f"Best performing method: {best_method}")
    print(f"Worst performing method: {worst_method}")
    print(f"Performance improvement: {improvement:.2f}%")
    print("\nMethod Characteristics:")
    print("- Lagrange: Traditional polynomial interpolation, can have oscillations")
    print("- Euler Method: First-order differential equation solver, models drift and volatility")
    print("- Runge-Kutta 4th: High-accuracy differential equation solver with mean reversion")


def generate_synthetic_data(prices, method_name, interpolation_func):
    """
    Generate synthetic data using interpolation methods.
    - Use first 50 real prices
    - Generate 25 synthetic prices (days 50-74)
    - Return combined 75 data points for training
    """
    # Use first 50 real values
    x_real = np.arange(50)
    y_real = prices[:50]
    
    # Generate 25 synthetic values (days 50-74)
    x_synthetic = np.arange(50, 75)
    
    try:
        y_synthetic = interpolation_func(x_real, y_real, x_synthetic)
        
        # More reasonable clipping bounds for the smaller dataset
        y_min, y_max = np.min(y_real), np.max(y_real)
        price_range = y_max - y_min
        lower_bound = y_min - 0.3 * price_range  # Tighter bounds
        upper_bound = y_max + 0.3 * price_range  # Tighter bounds
        
        y_synthetic = np.clip(y_synthetic, lower_bound, upper_bound)
        
        print(f"  Generated synthetic data for {method_name}")
        print(f"  Original range: [{y_min:.2f}, {y_max:.2f}]")
        print(f"  Synthetic range: [{np.min(y_synthetic):.2f}, {np.max(y_synthetic):.2f}]")
        print(f"  Clipping bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
    except Exception as e:
        print(f"  Warning: {method_name} failed, using linear interpolation fallback")
        print(f"  Error: {str(e)}")
        y_synthetic = np.interp(x_synthetic, x_real, y_real)
    
    # Combine real and synthetic data for training (75 points total)
    x_combined = np.arange(75)
    y_combined = np.concatenate([y_real, y_synthetic])
    
    return x_combined, y_combined

def train_and_predict(prices, x_train, y_train, model):
    """
    Train the model on 75 points and predict the remaining 25 points.
    """
    # Train model on 75 points (50 real + 25 synthetic)
    model.train(x_train, y_train)
    
    # Test on remaining 25 points (days 75-99)
    remaining_days = len(prices) - 75
    if remaining_days <= 0:
        print(f"Warning: Not enough data for prediction phase")
        return np.array([]), np.array([]), np.array([]), 0, 0, 0
    
    x_test = np.arange(75, 75 + remaining_days)
    y_pred = model.predict(x_test)
    y_actual = prices[75:75 + remaining_days]
    
    mse, mae, r2 = model.evaluate(y_actual, y_pred)
    return y_pred, x_test, y_actual, mse, mae, r2


if __name__ == "__main__":
    main()

