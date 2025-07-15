"""
Numerical Methods Stock Price Prediction Project

# This project shows how to use three different interpolation methods to generate fake (synthetic) stock prices.
# We use the first 50 real prices, then create 50 synthetic prices using each method.
# After that, we train a simple machine learning model (LinearRegression) to predict the next 150 prices.
# The three interpolation methods are:
# 1. Lagrange interpolation (uses all points to fit a curve)
# 2. Newton's divided difference interpolation (good for adding new points)
# 3. Newton's forward difference (best for equally spaced data and extrapolation)
"""


# Import modules
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from modules.interpolation_methods import InterpolationMethods
from modules.data_handler import DataHandler
from modules.model import StockModel
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
    days = 200
    data_handler = DataHandler(ticker=ticker, days=days)
    interpolation = InterpolationMethods()
    model = StockModel()
    viz = Visualization()

    # Fetch data
    if not data_handler.fetch_stock_data():
        return
    prices = data_handler.get_prices()

    results = {}
    # Method 1: Lagrange interpolation
    print("\n1. Running Lagrange interpolation...")
    x_train, y_train = generate_synthetic_data(prices, "Lagrange", interpolation.lagrange_interpolation)
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

    # Method 2: Newton's divided difference
    print("2. Running Newton's divided difference...")
    x_train, y_train = generate_synthetic_data(prices, "Newton Divided Diff", interpolation.newton_divided_difference)
    y_pred, x_test, y_actual, mse, mae, r2 = train_and_predict(prices, x_train, y_train, model)
    results["Newton Divided Diff"] = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_pred': y_pred,
        'y_actual': y_actual,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    # Method 3: Newton's forward difference
    print("3. Running Newton's forward difference...")
    x_train, y_train = generate_synthetic_data(prices, "Newton Forward Diff", interpolation.newton_forward_difference)
    y_pred, x_test, y_actual, mse, mae, r2 = train_and_predict(prices, x_train, y_train, model)
    results["Newton Forward Diff"] = {
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
    print("- Lagrange: Global polynomial, can have oscillations")
    print("- Newton Divided Diff: Efficient for adding new points")
    print("- Newton Forward Diff: Better for extrapolation with equal spacing")


def generate_synthetic_data(prices, method_name, interpolation_func):
    # Use first 50 real values
    x_real = np.arange(50)
    y_real = prices[:50]
    x_synthetic = np.arange(50, 100)
    try:
        y_synthetic = interpolation_func(x_real, y_real, x_synthetic)
        y_min, y_max = np.min(y_real), np.max(y_real)
        price_range = y_max - y_min
        lower_bound = y_min - 0.5 * price_range
        upper_bound = y_max + 0.5 * price_range
        y_synthetic = np.clip(y_synthetic, lower_bound, upper_bound)
        print(f"  Generated synthetic data for {method_name}")
        print(f"  Original range: [{y_min:.2f}, {y_max:.2f}]")
        print(f"  Synthetic range: [{np.min(y_synthetic):.2f}, {np.max(y_synthetic):.2f}]")
    except Exception as e:
        print(f"  Warning: {method_name} failed, using linear interpolation fallback")
        y_synthetic = np.interp(x_synthetic, x_real, y_real)
    x_combined = np.arange(100)
    y_combined = np.concatenate([y_real, y_synthetic])
    return x_combined, y_combined

def train_and_predict(prices, x_train, y_train, model):
    model.train(x_train, y_train)
    remaining_days = len(prices) - 100
    if remaining_days <= 0:
        print(f"Warning: Not enough data for prediction phase")
        return np.array([]), np.array([]), np.array([]), 0, 0, 0
    x_test = np.arange(100, 100 + remaining_days)
    y_pred = model.predict(x_test)
    y_actual = prices[100:100 + remaining_days]
    mse, mae, r2 = model.evaluate(y_actual, y_pred)
    return y_pred, x_test, y_actual, mse, mae, r2


if __name__ == "__main__":
    main()

