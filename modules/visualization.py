import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    # This class handles all plotting and visualization for gap-filling analysis
    # Creates comprehensive charts showing interpolation and prediction results
    def plot_comparison(self, prices, results, ticker):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Stock Price Prediction Comparison - {ticker}', fontsize=16)
        x_actual = np.arange(len(prices))
        axes[0, 0].plot(x_actual, prices, 'k-', linewidth=2, label='Actual Prices', alpha=0.7)
        colors = ['red', 'blue', 'green']
        for i, (method, res) in enumerate(results.items()):
            axes[0, 0].plot(res['x_train'], res['y_train'], color=colors[i], linestyle='--', alpha=0.6, label=f'{method} Training')
            axes[0, 0].plot(res['x_test'], res['y_pred'], color=colors[i], linewidth=2, label=f'{method} Prediction')
        axes[0, 0].axvline(x=50, color='orange', linestyle=':', alpha=0.7, label='Real/Synthetic Split')
        axes[0, 0].axvline(x=100, color='purple', linestyle=':', alpha=0.7, label='Train/Test Split')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Price Predictions Comparison')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        methods = list(results.keys())
        mse_values = [results[m]['mse'] for m in methods]
        bars = axes[0, 1].bar(methods, mse_values, color=['red', 'blue', 'green'], alpha=0.7)
        axes[0, 1].set_ylabel('Mean Squared Error')
        axes[0, 1].set_title('MSE Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, mse_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, f'{value:.2f}', ha='center', va='bottom')
        r2_values = [results[m]['r2'] for m in methods]
        bars = axes[1, 0].bar(methods, r2_values, color=['red', 'blue', 'green'], alpha=0.7)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('R² Score Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, r2_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, f'{value:.3f}', ha='center', va='bottom')
        best_method = min(results.keys(), key=lambda x: results[x]['mse'])
        best_results = results[best_method]
        residuals = best_results['y_actual'] - best_results['y_pred']
        axes[1, 1].scatter(best_results['x_test'], residuals, alpha=0.6, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title(f'Residuals - {best_method} (Best Method)')
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_detailed(self, prices, results, ticker):
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Detailed Interpolation Analysis - {ticker}', fontsize=16)
        colors = ['red', 'blue', 'green']
        for i, (method, res) in enumerate(results.items()):
            ax = axes[i]
            x_actual = np.arange(len(prices))
            ax.plot(x_actual, prices, 'k-', linewidth=2, label='Actual Prices', alpha=0.7)
            ax.scatter(np.arange(50), prices[:50], color='orange', s=30, label='Original 50 Points', zorder=5)
            synthetic_x = np.arange(50, 100)
            synthetic_y = res['y_train'][50:]
            ax.plot(synthetic_x, synthetic_y, color=colors[i], linestyle='--', linewidth=2, label=f'{method} Synthetic', alpha=0.8)
            ax.plot(res['x_test'], res['y_pred'], color=colors[i], linewidth=2, label=f'{method} Prediction')
            ax.axvline(x=50, color='orange', linestyle=':', alpha=0.7, label='Interpolation Start')
            ax.axvline(x=100, color='purple', linestyle=':', alpha=0.7, label='Prediction Start')
            ax.text(0.02, 0.98, f'MSE: {res["mse"]:.4f}\nMAE: {res["mae"]:.4f}\nR²: {res["r2"]:.4f}', transform=ax.transAxes, verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_xlabel('Days')
            ax.set_ylabel('Price ($)')
            ax.set_title(f'{method} Method')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_gap_filling_comparison(self, full_data, x_known, y_known, results, ticker, interpolation_indices, prediction_indices):
        """
        Plot the gap filling comparison showing:
        - Original data with gaps
        - Interpolated values
        - Model predictions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Gap Filling Analysis - {ticker}', fontsize=16)
        
        # Main comparison plot
        x_full = np.arange(len(full_data))
        axes[0, 0].plot(x_full, full_data, 'k-', linewidth=2, label='True Values', alpha=0.7)
        axes[0, 0].scatter(x_known, y_known, color='blue', s=40, label='Known Data', zorder=5)
        axes[0, 0].scatter(interpolation_indices, full_data[interpolation_indices], color='orange', s=40, label='Interpolation Target', zorder=5)
        axes[0, 0].scatter(prediction_indices, full_data[prediction_indices], color='red', s=40, label='Prediction Target', zorder=5)
        
        colors = ['red', 'blue', 'green']
        for i, (method, res) in enumerate(results.items()):
            # Plot interpolated values (part of training data)
            interp_y = res['y_train'][len(x_known):]  # Get interpolated values
            axes[0, 0].plot(interpolation_indices, interp_y, color=colors[i], marker='o', linestyle='--', alpha=0.7, label=f'{method} Interpolation')
            
            # Plot model predictions
            axes[0, 0].plot(prediction_indices, res['y_pred'], color=colors[i], marker='s', linestyle=':', linewidth=2, label=f'{method} Prediction')
        
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Gap Filling Comparison')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE comparison
        methods = list(results.keys())
        mse_values = [results[m]['mse'] for m in methods]
        bars = axes[0, 1].bar(methods, mse_values, color=['red', 'blue', 'green'], alpha=0.7)
        axes[0, 1].set_ylabel('Mean Squared Error')
        axes[0, 1].set_title('MSE Comparison (Model Predictions)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, mse_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, f'{value:.2f}', ha='center', va='bottom')
        
        # R² comparison
        r2_values = [results[m]['r2'] for m in methods]
        bars = axes[1, 0].bar(methods, r2_values, color=['red', 'blue', 'green'], alpha=0.7)
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].set_title('R² Score Comparison (Model Predictions)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, r2_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, f'{value:.3f}', ha='center', va='bottom')
        
        # Error analysis for best method
        best_method = min(results.keys(), key=lambda x: results[x]['mse'])
        best_results = results[best_method]
        
        # Plot interpolation errors
        interp_errors = []
        interp_y = best_results['y_train'][len(x_known):]
        for i, idx in enumerate(interpolation_indices):
            error = abs(interp_y[i] - full_data[idx])
            interp_errors.append(error)
        
        # Plot prediction errors
        pred_errors = []
        for i, idx in enumerate(prediction_indices):
            error = abs(best_results['y_pred'][i] - full_data[idx])
            pred_errors.append(error)
        
        axes[1, 1].scatter(interpolation_indices, interp_errors, color='orange', alpha=0.6, label='Interpolation Errors', s=40)
        axes[1, 1].scatter(prediction_indices, pred_errors, color='red', alpha=0.6, label='Prediction Errors', s=40)
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title(f'Error Analysis - {best_method} (Best Method)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
