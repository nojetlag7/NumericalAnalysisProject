import numpy as np
import matplotlib.pyplot as plt
from modules.numerical_methods import NumericalMethods
from modules.data_handler import DataHandler
from modules.stock_model import StockModel

def focused_comparison():
    """
    Detailed comparison of the top 2 performing methods: RK4 vs Euler
    with extended prediction horizons and deeper analysis
    """
    print("\n" + "="*60)
    print("FOCUSED COMPARISON: RUNGE-KUTTA vs EULER")
    print("="*60)
    
    # Load data
    data_handler = DataHandler(ticker='AAPL', days=150)
    data_handler.fetch_stock_data()
    prices = data_handler.get_prices()
    
    # Test both methods with different prediction horizons
    horizons = [30, 50, 70]  # Different numbers of prediction points
    methods = {
        'Euler Method': NumericalMethods.euler_method,
        'Runge-Kutta 4th': NumericalMethods.runge_kutta_4th_order
    }
    
    results = {}
    
    for horizon in horizons:
        print(f"\nTesting {horizon}-point prediction horizon:")
        results[horizon] = {}
        
        for method_name, method_func in methods.items():
            # Use first 50 points for training, predict next 'horizon' points
            x_real = np.arange(50)
            y_real = prices[:50]
            x_target = np.arange(50, 50 + horizon)
            
            try:
                y_synthetic = method_func(x_real, y_real, x_target)
                
                # Train model on synthetic data
                model = StockModel()
                model.train(x_target, y_synthetic)
                
                # Predict the remaining points
                start_idx = 50 + horizon
                remaining_points = min(30, len(prices) - start_idx)  # Predict up to 30 more points
                
                if remaining_points > 0:
                    x_test = np.arange(start_idx, start_idx + remaining_points)
                    y_pred = model.predict(x_test)
                    y_actual = prices[start_idx:start_idx + remaining_points]
                    
                    mse, mae, r2 = model.evaluate(y_actual, y_pred)
                    
                    results[horizon][method_name] = {
                        'synthetic': y_synthetic,
                        'x_test': x_test,
                        'y_pred': y_pred,
                        'y_actual': y_actual,
                        'mse': mse,
                        'mae': mae,
                        'r2': r2
                    }
                    
                    print(f"  {method_name:20} MSE: {mse:8.2f}, MAE: {mae:6.2f}, R²: {r2:7.4f}")
                    
            except Exception as e:
                print(f"  {method_name:20} Failed: {str(e)}")
    
    # Performance comparison table
    print(f"\n{'PERFORMANCE SUMMARY'}")
    print("="*60)
    print(f"{'Horizon':<10} {'Method':<20} {'MSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 60)
    
    for horizon in horizons:
        for method_name in methods.keys():
            if method_name in results[horizon]:
                res = results[horizon][method_name]
                print(f"{horizon:<10} {method_name:<20} {res['mse']:<10.2f} {res['mae']:<10.2f} {res['r2']:<10.4f}")
    
    # Stability analysis - check how performance changes with prediction horizon
    print(f"\n{'STABILITY ANALYSIS'}")
    print("="*60)
    for method_name in methods.keys():
        print(f"\n{method_name}:")
        mse_values = []
        for horizon in horizons:
            if method_name in results[horizon]:
                mse_val = results[horizon][method_name]['mse']
                mse_values.append(mse_val)
                print(f"  {horizon:2d}-point horizon: MSE = {mse_val:8.2f}")
        
        if len(mse_values) > 1:
            mse_stability = np.std(mse_values) / np.mean(mse_values) if np.mean(mse_values) > 0 else 0
            print(f"  Stability (CV): {mse_stability:.4f} (lower is more stable)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detailed RK4 vs Euler Comparison', fontsize=16)
    
    # Plot 1: Performance vs Horizon
    horizon_data = {method: [] for method in methods.keys()}
    for horizon in horizons:
        for method_name in methods.keys():
            if method_name in results[horizon]:
                horizon_data[method_name].append(results[horizon][method_name]['mse'])
            else:
                horizon_data[method_name].append(np.nan)
    
    for method_name, mse_vals in horizon_data.items():
        color = 'blue' if 'Euler' in method_name else 'green'
        axes[0, 0].plot(horizons, mse_vals, 'o-', color=color, label=method_name, linewidth=2, markersize=8)
    
    axes[0, 0].set_xlabel('Prediction Horizon (points)')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Performance vs Prediction Horizon')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Sample prediction comparison (70-point horizon)
    if 70 in results and len(results[70]) >= 2:
        for i, (method_name, method_data) in enumerate(results[70].items()):
            color = 'blue' if 'Euler' in method_name else 'green'
            
            # Plot actual vs predicted
            axes[0, 1].plot(method_data['x_test'], method_data['y_actual'], 'k-', 
                           linewidth=2, label='Actual' if i == 0 else "", alpha=0.7)
            axes[0, 1].plot(method_data['x_test'], method_data['y_pred'], 
                           color=color, linewidth=2, label=f'{method_name} Prediction')
        
        axes[0, 1].set_xlabel('Time (days)')
        axes[0, 1].set_ylabel('Price')
        axes[0, 1].set_title('70-Point Prediction Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution comparison (70-point horizon)
    if 70 in results and len(results[70]) >= 2:
        errors_data = []
        labels = []
        for method_name, method_data in results[70].items():
            errors = method_data['y_actual'] - method_data['y_pred']
            errors_data.append(errors)
            labels.append(method_name)
        
        axes[1, 0].boxplot(errors_data, labels=labels)
        axes[1, 0].set_ylabel('Prediction Error')
        axes[1, 0].set_title('Error Distribution (70-point prediction)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Method comparison bar chart
    if 70 in results:
        methods_list = list(results[70].keys())
        mse_values = [results[70][m]['mse'] for m in methods_list]
        mae_values = [results[70][m]['mae'] for m in methods_list]
        
        x_pos = np.arange(len(methods_list))
        width = 0.35
        
        bars1 = axes[1, 1].bar(x_pos - width/2, mse_values, width, label='MSE', alpha=0.7, color='lightblue')
        axes1_twin = axes[1, 1].twinx()
        bars2 = axes1_twin.bar(x_pos + width/2, mae_values, width, label='MAE', alpha=0.7, color='lightcoral')
        
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('MSE', color='blue')
        axes1_twin.set_ylabel('MAE', color='red')
        axes[1, 1].set_title('MSE vs MAE Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([m.replace(' Method', '').replace(' 4th', '') for m in methods_list])
        
        # Add value labels on bars
        for bar, value in zip(bars1, mse_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                           f'{value:.1f}', ha='center', va='bottom')
        for bar, value in zip(bars2, mae_values):
            axes1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                           f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('rk4_vs_euler_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final recommendation
    print(f"\n{'RECOMMENDATION'}")
    print("="*60)
    
    # Calculate average performance across horizons
    avg_performance = {}
    for method_name in methods.keys():
        mse_values = []
        for horizon in horizons:
            if method_name in results[horizon]:
                mse_values.append(results[horizon][method_name]['mse'])
        
        if mse_values:
            avg_performance[method_name] = {
                'avg_mse': np.mean(mse_values),
                'stability': np.std(mse_values) / np.mean(mse_values) if np.mean(mse_values) > 0 else 0
            }
    
    if avg_performance:
        best_method = min(avg_performance.keys(), key=lambda x: avg_performance[x]['avg_mse'])
        most_stable = min(avg_performance.keys(), key=lambda x: avg_performance[x]['stability'])
        
        print(f"Best Average Performance: {best_method}")
        print(f"Most Stable Performance:  {most_stable}")
        
        for method, perf in avg_performance.items():
            improvement_vs_other = 0
            other_methods = [m for m in avg_performance.keys() if m != method]
            if other_methods:
                other_avg = np.mean([avg_performance[m]['avg_mse'] for m in other_methods])
                improvement_vs_other = ((other_avg - perf['avg_mse']) / other_avg) * 100
            
            print(f"\n{method}:")
            print(f"  Average MSE: {perf['avg_mse']:.2f}")
            print(f"  Stability:   {perf['stability']:.4f}")
            print(f"  Improvement: {improvement_vs_other:+.2f}% vs other methods")

if __name__ == "__main__":
    focused_comparison()
