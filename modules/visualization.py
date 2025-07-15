import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    # This class handles all plotting and visualization
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
