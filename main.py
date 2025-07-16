"""
Numerical Methods Stock Price Prediction Project

# This project shows how to use three different interpolation methods to fill gaps in stock price data.
# We use 200 real prices, randomly remove 100 values, then:
# 1. Use interpolation methods to fill 50 of the missing values
# 2. Train ML models to predict the remaining 50 missing values
# 3. Compare model predictions against true values
# The three interpolation methods are:
# 1. Lagrange interpolation (uses all points to fit a curve)
# 2. Newton's divided difference interpolation (good for adding new points)
# 3. Cubic Spline interpolation (smooth piecewise polynomials, ideal for gap-filling)
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
    # Main function to run the gap-filling analysis project
    # It sets up the modules, creates gaps in data, uses interpolation to fill some gaps,
    # trains models to predict remaining gaps, and compares results
    # 
    # Process:
    # 1. Load 200 stock prices
    # 2. Remove 100 random values (50 for interpolation, 50 for prediction)
    # 3. Use interpolation methods to fill 50 gaps
    # 4. Train models on known + interpolated data (150 points)
    # 5. Predict the remaining 50 missing values
    # 6. Compare predictions against true values
    """
    print("Numerical Methods Stock Price Prediction")
    print("="*50)

    # Set up modules
    ticker = 'AAPL'
    days = 200  # Use first 200 days instead of 100
    data_handler = DataHandler(ticker=ticker, days=days)
    interpolation = InterpolationMethods()
    model = StockModel()
    viz = Visualization()

    # Fetch data
    if not data_handler.fetch_stock_data():
        return
    prices = data_handler.get_prices()
    
    # Use first 200 prices as our complete dataset
    full_data = prices[:200]
    x_full = np.arange(200)
    
    # Create gaps: randomly remove 100 values (doubled from 50)
    np.random.seed(42)  # For reproducible results
    missing_indices = np.random.choice(range(10, 190), size=100, replace=False)  # Avoid edges
    missing_indices = np.sort(missing_indices)
    
    # Split missing indices: 50 for interpolation, 50 for model prediction (doubled from 25 each)
    interpolation_indices = missing_indices[:50]
    prediction_indices = missing_indices[50:]
    
    # Create the known data (without the missing values)
    known_indices = np.setdiff1d(x_full, missing_indices)
    x_known = known_indices
    y_known = full_data[known_indices]
    
    print(f"Original dataset size: {len(full_data)}")
    print(f"Known data points: {len(known_indices)}")
    print(f"Missing points to interpolate: {len(interpolation_indices)}")
    print(f"Missing points for model prediction: {len(prediction_indices)}")

    results = {}
    
    # Method 1: Lagrange interpolation
    print("\n1. Running Lagrange interpolation...")
    x_train, y_train, model_pred, model_actual = process_method(
        full_data, x_known, y_known, interpolation_indices, prediction_indices,
        "Lagrange", interpolation.lagrange_interpolation, model
    )
    mse, mae, r2 = model.evaluate(model_actual, model_pred)
    results["Lagrange"] = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': prediction_indices,
        'y_pred': model_pred,
        'y_actual': model_actual,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    # Method 2: Newton's divided difference
    print("2. Running Newton's divided difference...")
    x_train, y_train, model_pred, model_actual = process_method(
        full_data, x_known, y_known, interpolation_indices, prediction_indices,
        "Newton Divided Diff", interpolation.newton_divided_difference, model
    )
    mse, mae, r2 = model.evaluate(model_actual, model_pred)
    results["Newton Divided Diff"] = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': prediction_indices,
        'y_pred': model_pred,
        'y_actual': model_actual,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

    # Method 3: Cubic Spline interpolation
    print("3. Running Cubic Spline interpolation...")
    x_train, y_train, model_pred, model_actual = process_method(
        full_data, x_known, y_known, interpolation_indices, prediction_indices,
        "Cubic Spline", interpolation.cubic_spline_interpolation, model
    )
    mse, mae, r2 = model.evaluate(model_actual, model_pred)
    results["Cubic Spline"] = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': prediction_indices,
        'y_pred': model_pred,
        'y_actual': model_actual,
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
    try:
        viz.plot_gap_filling_comparison(full_data, x_known, y_known, results, ticker, interpolation_indices, prediction_indices)
    except Exception as e:
        print(f"Warning: Visualization failed ({str(e)})")
        print("This might be due to missing matplotlib display backend")
        # Try to show basic plot info instead
        try:
            import matplotlib.pyplot as plt
            print("Matplotlib is available, but display may not be configured properly")
        except ImportError:
            print("Matplotlib is not available for plotting")

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
    print("- Cubic Spline: Smooth piecewise polynomials, ideal for interpolation")
    print(f"\nData Processing Summary:")
    print(f"- Original data points: 200")
    print(f"- Missing data points: 100")
    print(f"- Points filled by interpolation: 50")
    print(f"- Points predicted by models: 50")
    print(f"- Remaining known points: 100")


def process_method(full_data, x_known, y_known, interpolation_indices, prediction_indices, method_name, interpolation_func, model):
    """
    Process a single interpolation method for gap filling analysis:
    1. Use interpolation to fill 50 missing values from known data
    2. Train model on known + interpolated data (150 points total)
    3. Predict the remaining 50 missing values using the trained model
    4. Apply bounds checking and error handling for numerical stability
    
    Args:
        full_data: Complete dataset with 200 values
        x_known: Indices of 100 known data points
        y_known: Values of 100 known data points
        interpolation_indices: Indices of 50 values to interpolate
        prediction_indices: Indices of 50 values to predict with model
        method_name: Name of interpolation method for logging
        interpolation_func: Function to perform interpolation
        model: Machine learning model for prediction
    
    Returns:
        x_train: Training data indices (150 points)
        y_train: Training data values (150 points)
        model_pred: Model predictions for 50 test points
        model_actual: True values for 50 test points
    """
    try:
        # Use interpolation to fill the first set of missing values
        y_interpolated = interpolation_func(x_known, y_known, interpolation_indices)
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(y_interpolated)):
            print(f"  Warning: {method_name} produced invalid values, using linear interpolation fallback")
            y_interpolated = np.interp(interpolation_indices, x_known, y_known)
        
        # Apply reasonable bounds based on known data
        y_min, y_max = np.min(y_known), np.max(y_known)
        y_range = y_max - y_min
        lower_bound = y_min - 0.5 * y_range
        upper_bound = y_max + 0.5 * y_range
        y_interpolated = np.clip(y_interpolated, lower_bound, upper_bound)
        
        # Combine known data with interpolated data for training
        x_train = np.concatenate([x_known, interpolation_indices])
        y_train = np.concatenate([y_known, y_interpolated])
        
        # Sort training data by x values for consistency
        sort_idx = np.argsort(x_train)
        x_train = x_train[sort_idx]
        y_train = y_train[sort_idx]
        
        # Train model on known + interpolated data
        model.train(x_train, y_train)
        
        # Use model to predict the remaining missing values
        model_pred = model.predict(prediction_indices)
        model_actual = full_data[prediction_indices]
        
        # Calculate interpolation accuracy for reporting
        interp_actual = full_data[interpolation_indices]
        interp_mse = np.mean((y_interpolated - interp_actual) ** 2)
        
        print(f"  {method_name}: Interpolated {len(interpolation_indices)} values (MSE: {interp_mse:.4f})")
        print(f"  Training data size: {len(x_train)} points")
        print(f"  Interpolation range: [{np.min(y_interpolated):.2f}, {np.max(y_interpolated):.2f}]")
        
        return x_train, y_train, model_pred, model_actual
        
    except Exception as e:
        print(f"  Error: {method_name} failed ({str(e)}), using linear interpolation fallback")
        # Fallback to linear interpolation
        y_interpolated = np.interp(interpolation_indices, x_known, y_known)
        
        x_train = np.concatenate([x_known, interpolation_indices])
        y_train = np.concatenate([y_known, y_interpolated])
        
        sort_idx = np.argsort(x_train)
        x_train = x_train[sort_idx]
        y_train = y_train[sort_idx]
        
        model.train(x_train, y_train)
        model_pred = model.predict(prediction_indices)
        model_actual = full_data[prediction_indices]
        
        return x_train, y_train, model_pred, model_actual


if __name__ == "__main__":
    main()

