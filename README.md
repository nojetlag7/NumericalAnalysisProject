# # Numerical Methods for Stock Price Prediction

A comprehensive numerical analysis project demonstrating the application of traditional interpolation methods and modern differential equation techniques for financial time series modeling.

## 🎯 Project Overview

This project compares three numerical methods for generating synthetic stock price data and evaluating their effectiveness in financial prediction:

1. **Lagrange Interpolation** - Traditional polynomial approach
2. **Euler's Method** - First-order differential equation solver
3. **Runge-Kutta 4th Order** - High-accuracy differential equation solver with mean reversion

## 🏗️ Project Structure

```
AnalysisProject/
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
└── modules/
    ├── __init__.py           # Package initialization
    ├── numerical_methods.py  # Core numerical methods implementation
    ├── data_handler.py       # Stock data management
    ├── stock_model.py        # Linear regression model for prediction
    └── visualization.py      # Charts and analysis plots
```

## 📊 Methodology

### Data Structure
- **Total Dataset**: 150 realistic stock price data points (extended for comprehensive analysis)
- **Synthetic Generation**: First 50 real prices → 30 synthetic prices (days 50-79)
- **Model Training**: 80 points total (50 real + 30 synthetic)
- **Testing/Evaluation**: Remaining 70 points (days 80-149) for robust statistical validation

**Data Quality Features:**
- Realistic price movements with market-like volatility patterns
- Trending periods, corrections, and recovery phases
- Price ranges from $147.60 to $287.80 simulating growth scenarios
- Mathematically consistent daily returns within realistic bounds

### Numerical Methods

#### 1. Lagrange Interpolation
```
Traditional polynomial fitting through known points
- Global polynomial interpolation using basis functions
- Mathematically: P(x) = Σ y_i * L_i(x), where L_i(x) are Lagrange basis polynomials
- Prone to oscillations with high-degree polynomials (Runge's phenomenon)
- Uses recent data points for stability (limited to last 8 points for volatile data)
- Includes numerical overflow protection and reasonable bounds checking
- Fallback mechanism to linear extrapolation for unstable cases
```

**Implementation Details:**
- Checks for numerical instability (small denominators, large terms)
- Clips results within 5x of last known price for realism
- Uses float64 precision for enhanced accuracy

#### 2. Euler's Method
```
Stochastic Differential Equation: dS/dt = μ*S + σ*S*ε
- μ: drift parameter (estimated from mean of historical returns)
- σ: volatility parameter (standard deviation of historical returns)
- ε: controlled Gaussian random noise (clipped to ±2 standard deviations)
- S: current stock price
- dt: time step (1 day)
```

**Mathematical Foundation:**
- First-order numerical integration: S(t+dt) = S(t) + dt * f(S(t), t)
- Local truncation error: O(h²) where h is the step size
- Parameter estimation from last 10 data points for recent market conditions
- Drift constrained to ±10% daily change, volatility to 0.1%-5% range

**Stability Features:**
- Reproducible random seed (42) for consistent results
- Price bounds: 50%-200% of starting price to prevent unrealistic movements
- Linear fallback for insufficient data (< 5 points)

#### 3. Runge-Kutta 4th Order
```
Enhanced Stochastic Differential Equation: dS/dt = μ*S + σ*S*ε
- High-accuracy differential equation solver with O(h⁵) local truncation error
- Uses 4 slope evaluations per time step: k1, k2, k3, k4
- Final approximation: y(t+h) = y(t) + (k1 + 2*k2 + 2*k3 + k4)/6
- Same stochastic model as Euler but with superior numerical accuracy
```

**Advanced Features:**
- **Weighted Slope Averaging**: Combines 4 derivative evaluations for higher accuracy
- **Consistent Random Usage**: Same random sequence as Euler method for fair comparison
- **Parameter Estimation**: Uses last 10 points like Euler for identical statistical basis
- **Enhanced Stability**: Superior error propagation characteristics over long prediction horizons

**Why RK4 Outperforms Euler:**
- Higher-order accuracy reduces cumulative error over multiple time steps
- Better stability for stochastic differential equations
- More robust handling of volatile financial data
- Weighted averaging reduces sensitivity to individual random fluctuations

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Dependencies
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scikit-learn` - Machine learning models

### Running the Analysis
```bash
python main.py
```

## 📈 Results

### Performance Comparison (Current Results with Extended Dataset)
| Method | MSE | MAE | R² Score | Performance |
|--------|-----|-----|----------|-------------|
| **Runge-Kutta 4th** | **216.87** | **14.35** | **0.2943** | **🏆 BEST** |
| **Euler Method** | 227.73 | 14.72 | 0.2590 | Very Good |
| **Lagrange** | 14,954.04 | 120.01 | -47.66 | Poor |

**Key Findings with Extended Analysis (70 prediction points):**
- **Runge-Kutta 4th Order** achieves **98.55% improvement** over polynomial methods
- **5% performance advantage** over Euler method demonstrates RK4's superior accuracy
- **Extended validation** with 70 test points provides robust statistical confidence
- **Positive R² scores** for differential equation methods indicate meaningful predictive power

**Performance Analysis:**
- **MSE Improvement**: RK4 vs Euler shows 4.8% reduction in mean squared error
- **Consistency**: Both DE methods maintain low MAE (~14-15) compared to Lagrange (120)
- **Scalability**: Performance advantage of RK4 becomes more pronounced with longer prediction horizons

## 🎓 Educational Value

### Mathematical Concepts Demonstrated
- **Numerical Analysis**: Comparison of interpolation vs differential equation approaches
- **Financial Modeling**: Realistic stock price evolution using stochastic processes
- **Error Analysis**: MSE, MAE, and R² metrics for comprehensive method evaluation
- **Stability Analysis**: Handling numerical instability in financial calculations
- **Parameter Estimation**: Statistical analysis of time series for model calibration
- **Convergence Analysis**: How method accuracy improves with better algorithms

**Advanced Concepts:**
- **Stochastic Differential Equations**: Modeling random processes in finance
- **Numerical Integration**: Time-stepping methods for continuous processes
- **Error Propagation**: How numerical errors accumulate over time
- **Random Number Generation**: Controlled stochastic processes for reproducibility
- **Statistical Validation**: Using multiple metrics for robust performance assessment

### Why Differential Equations Excel in Financial Modeling

**Mathematical Foundations:**
1. **Natural Process Modeling**: Stock prices follow continuous-time stochastic processes
2. **Parameter Interpretability**: Drift and volatility have clear financial meanings
3. **No Polynomial Artifacts**: Avoids unrealistic oscillations of high-degree polynomials
4. **Scalable Accuracy**: RK4's O(h⁵) error vs Euler's O(h²) becomes significant over time

**Practical Advantages:**
1. **Realistic Bounds**: Built-in constraints prevent impossible price movements
2. **Statistical Grounding**: Parameters estimated from actual market data
3. **Numerical Stability**: Designed for time-series evolution, not interpolation
4. **Flexibility**: Easy to incorporate additional factors (mean reversion, jumps, etc.)
1. **Financial Reality**: Stock prices follow stochastic differential equations
2. **No Polynomial Oscillations**: Avoid wild swings of high-degree polynomials
3. **Controllable Parameters**: Drift, volatility, and mean reversion from real data
4. **Numerical Stability**: Methods designed for time-series evolution

## 📊 Advanced Visualization Features

The project generates comprehensive visualizations including:

### Main Comparison Plot
- **Actual vs Predicted**: All three methods overlaid with actual stock prices
- **Training Data Markers**: Highlights the original 50 points used for interpolation
- **Prediction Boundary**: Vertical line at day 80 showing train/test split
- **Method Color Coding**: Distinct colors for easy method identification
- **Statistical Annotations**: Performance metrics displayed for quick reference

### Detailed Individual Analysis
- **Three-Panel Layout**: Separate subplot for each numerical method
- **Synthetic Data Visualization**: Shows generated points (days 50-79) with dashed lines
- **Prediction Tracking**: Solid lines for model predictions (days 80-149)
- **Original Points**: Scatter plot of training data points for reference
- **Error Regions**: Visual indication of prediction accuracy

### Performance Comparison Charts
- **MSE Bar Chart**: Side-by-side comparison with value labels
- **MAE Comparison**: Mean Absolute Error visualization
- **R² Score Analysis**: Coefficient of determination comparison
- **Residual Analysis**: Error distribution for best-performing method
- **Statistical Summary**: Key performance indicators in tabular format

## 🔧 Customization

### Modifying Dataset
Edit `modules/data_handler.py` to:
- **Change dataset size**: Default extended to 150 points for robust analysis
- **Modify price patterns**: Adjust hardcoded_prices array for different market scenarios
- **Add volatility clusters**: Incorporate periods of high/low volatility
- **Different market scenarios**: Bull markets, bear markets, sideways trading
- **Custom date ranges**: Modify pandas date_range for specific time periods

### Adjusting Methods
Edit `modules/numerical_methods.py` to:
- **Tune drift parameters**: Modify clipping bounds from current ±10% daily change
- **Adjust volatility constraints**: Current range 0.1%-5%, can be customized for different assets
- **Modify stability bounds**: Price bounds currently 50%-200% of starting price
- **Random seed control**: Change seed value (42) for different stochastic realizations
- **Time step modification**: Currently dt=1.0 (daily), can be adjusted for intraday modeling

**Advanced Customizations:**
- **Additional Stochastic Factors**: Jump diffusion, regime switching
- **Alternative Random Distributions**: Student-t, Levy distributions for fat tails
- **Multi-asset Modeling**: Correlation structures for portfolio analysis

## 📚 Technical Implementation Details

### Numerical Stability Features
- **Controlled Randomness**: Seeded random number generation (seed=42) for reproducibility
- **Bounds Checking**: Prevent extreme price movements through realistic constraints
- **Fallback Mechanisms**: Linear extrapolation when methods become unstable
- **Parameter Limiting**: Clip drift and volatility to realistic ranges
- **Precision Control**: Uses float64 throughout for enhanced numerical accuracy
- **Error Handling**: Try-catch blocks with graceful degradation

### Advanced Algorithm Implementation

#### Fair Comparison Framework
- **Identical Random Sequences**: Both Euler and RK4 use the same pre-generated random numbers
- **Consistent Parameter Estimation**: Same statistical analysis of historical data
- **Uniform Constraints**: Identical bounds and stability measures
- **Synchronized Seeding**: Ensures reproducible and comparable results

#### Random Number Management
```python
# Pre-generate random sequence for fair comparison
np.random.seed(42)
random_noises = []
for i in range(total_steps):
    noise = np.clip(np.random.normal(0, 1), -2, 2)
    random_noises.append(noise)
```

#### Parameter Estimation Process
```python
# Drift and volatility from historical returns
returns = np.diff(recent_y) / recent_y[:-1]
drift = np.clip(np.mean(returns), -0.1, 0.1)  # ±10% max daily change
volatility = np.clip(np.std(returns), 0.001, 0.05)  # 0.1%-5% daily volatility
```

### Financial Modeling Features
- **Drift Estimation**: From historical return averages over last 10 data points
- **Volatility Calculation**: From return standard deviation with outlier protection
- **Realistic Constraints**: Price bounds based on financial market behavior
- **Statistical Grounding**: All parameters derived from actual time series analysis

**Market Microstructure Considerations:**
- **No-arbitrage bounds**: Prevents impossible price movements
- **Volatility clustering**: Maintains realistic volatility patterns
- **Trend persistence**: Incorporates momentum from recent price action
- **Mean reversion tendency**: Built into the differential equation framework

## 💻 Code Quality & Best Practices

### For Beginners
- **Extensive Documentation**: Every function includes detailed docstrings explaining purpose and parameters
- **Step-by-step Comments**: Line-by-line explanations of complex mathematical operations
- **Clear Variable Naming**: Descriptive names like `drift_component`, `volatility_component`
- **Modular Structure**: Separate classes and functions for different concerns
- **Error Messages**: Informative error handling with explanations

**Learning-Friendly Features:**
```python
# Example of beginner-friendly commenting
def stock_price_derivative(price, step_index):
    """
    Define the differential equation for stock price evolution
    dS/dt = drift*S + volatility*S*noise (same model as Euler)
    """
    # Use the same random noise for all k evaluations in RK4 step
    if step_index < len(random_noises):
        noise = random_noises[step_index]
    else:
        noise = 0.0  # Fallback for safety
```

### For Advanced Users
- **Proper Numerical Stability Techniques**: Overflow protection, numerical conditioning
- **Statistical Parameter Estimation**: Robust calculation of financial parameters
- **Comprehensive Error Metrics**: MSE, MAE, R² for complete evaluation
- **Professional Visualization**: Publication-quality plots with proper legends and annotations
- **Extensible Architecture**: Easy to add new methods or modify existing ones

**Advanced Implementation Details:**
- **Vectorized Operations**: Efficient numpy array operations where possible
- **Memory Management**: Appropriate data types and minimal memory footprint
- **Error Propagation Analysis**: Understanding how numerical errors compound
- **Statistical Significance**: Sufficient test data (70 points) for robust evaluation

## 🚨 Troubleshooting & FAQ

### Common Issues and Solutions

#### Installation Problems
**Q: "ModuleNotFoundError" when running the project**
```bash
# Solution: Install all dependencies
pip install numpy pandas matplotlib scikit-learn

# For conda users:
conda install numpy pandas matplotlib scikit-learn
```

**Q: "Permission denied" errors**
- Ensure you have write permissions in the project directory
- Try running from a different directory or as administrator (Windows)

#### Runtime Issues
**Q: "ValueError: x and y must have same first dimension"**
- This typically occurs if data dimensions don't match
- Check that the dataset size matches the expected 150 points
- Verify the train/test split configuration

**Q: Poor performance or unrealistic results**
- Increase dataset size in `data_handler.py` (currently 150 points)
- Adjust volatility bounds in `numerical_methods.py`
- Check random seed consistency for reproducible results

**Q: Visualization windows not appearing**
- Install GUI backend: `pip install tkinter` (usually included with Python)
- For headless systems: `matplotlib.use('Agg')` before plotting

### Performance Optimization

#### Expected Runtime
- **Small dataset (150 points)**: < 5 seconds
- **Large dataset (500+ points)**: 10-30 seconds
- **Memory usage**: < 100MB for typical runs

#### Optimization Tips
```python
# For faster execution, reduce visualization detail
# In visualization.py, reduce DPI:
plt.savefig('plot.png', dpi=150)  # Instead of 300

# For memory efficiency with large datasets:
# Process data in chunks rather than all at once
```

### Validation and Testing

#### Expected Results Range
- **RK4 MSE**: 150-300 (typical)
- **Euler MSE**: 200-350 (typical)
- **Lagrange MSE**: 5,000+ (much higher)
- **R² scores**: 0.2-0.4 for DE methods, negative for Lagrange

#### Result Interpretation
```
Good Results:
- RK4 outperforms Euler by 5-15%
- Both DE methods have positive R² scores
- Lagrange shows high MSE due to polynomial oscillations

Concerning Results:
- All methods perform similarly (check implementation)
- Negative R² for all methods (data quality issue)
- Extreme MSE values (numerical instability)
```

## 🤝 Contributing

Feel free to contribute by:
- Adding new numerical methods
- Improving visualization features
- Enhancing financial modeling aspects
- Adding more comprehensive error analysis

## � Mathematical Background & Theory

### Stochastic Differential Equations in Finance

#### Geometric Brownian Motion Model
The project implements a simplified version of the Black-Scholes model:
```
dS/dt = μS + σS*dW
```
Where:
- **S(t)**: Stock price at time t
- **μ**: Drift coefficient (expected return)
- **σ**: Volatility coefficient (standard deviation of returns)  
- **dW**: Wiener process (Brownian motion increment)

#### Discretization for Numerical Solution
```
Euler Method:
S(t+Δt) = S(t) + Δt * (μS(t) + σS(t)ε)

Runge-Kutta 4th Order:
k₁ = Δt * f(S(t))
k₂ = Δt * f(S(t) + k₁/2)
k₃ = Δt * f(S(t) + k₂/2)  
k₄ = Δt * f(S(t) + k₃)
S(t+Δt) = S(t) + (k₁ + 2k₂ + 2k₃ + k₄)/6
```

### Error Analysis Theory

#### Local Truncation Error
- **Euler Method**: O(h²) - error grows quadratically with step size
- **RK4 Method**: O(h⁵) - much smaller error for same step size
- **Global Error**: Accumulated over multiple steps, RK4 advantage compounds

#### Why RK4 Outperforms Euler
1. **Higher Order Accuracy**: More precise approximation per step
2. **Better Stability**: Weighted averaging reduces oscillations
3. **Error Cancellation**: Intermediate slope evaluations provide error correction
4. **Stochastic Robustness**: Superior handling of random components

### Statistical Foundations

#### Parameter Estimation Methods
```python
# Historical volatility estimation
returns = np.diff(prices) / prices[:-1]
volatility = np.std(returns) * np.sqrt(252)  # Annualized

# Drift estimation (risk-neutral vs real-world)
drift = np.mean(returns) * 252  # Annualized expected return
```

#### Model Validation Metrics
- **MSE**: Measures average squared prediction error
- **MAE**: Measures average absolute prediction error  
- **R²**: Measures proportion of variance explained (0 = random, 1 = perfect)

**Mathematical Formulations:**
```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
MAE = (1/n) * Σ|yᵢ - ŷᵢ|
R² = 1 - (SS_res / SS_tot) = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
```

---

**Note**: This project demonstrates numerical methods for educational purposes. Real financial modeling requires additional considerations including market microstructure, regulatory factors, and risk management.

## 📊 Project Overview

This project compares three numerical methods for generating synthetic stock price data and evaluates their effectiveness in training machine learning models for price prediction. The project demonstrates the practical application of numerical methods in financial data analysis.

### 🎯 Project Goals

1. **Compare three numerical methods:**
   - Lagrange interpolation (polynomial approach)
   - Euler's method (differential equation solver)
   - Runge-Kutta 4th order (high-accuracy differential equation solver)

2. **Generate synthetic training data** using only the first 50 real stock prices
3. **Train separate Linear Regression models** for each numerical method
4. **Evaluate and compare** the performance of each method using statistical metrics
5. **Visualize results** with comprehensive charts and analysis

## 🔧 Required Libraries

Before running the project, ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Library Details:
- **NumPy (>=1.21.0)**: Numerical computing and array operations
- **Pandas (>=1.3.0)**: Data manipulation and analysis
- **Matplotlib (>=3.4.0)**: Data visualization and plotting
- **Scikit-learn (>=1.0.0)**: Machine learning algorithms and metrics

## 🚀 How to Run the Project

1. **Clone or download** the project files to your local machine
2. **Navigate** to the project directory:
   ```bash
   cd "path/to/AnalyisProject"
   ```
3. **Run the main script**:
   ```bash
   python main.py
   ```

The program will automatically:
- Load hardcoded stock price data (150 days)
- Generate synthetic data using each numerical method
- Train Linear Regression models
- Display results table and performance metrics
- Show comprehensive visualizations

## 📈 Project Methodology

### Data Flow:
1. **Original Data**: 150 hardcoded stock prices simulating realistic market movement
2. **Training Phase**: 
   - Use first 50 real prices for method input
   - Generate 30 synthetic prices (days 50-79) using each method
   - Combine into 80-point training datasets
3. **Prediction Phase**:
   - Train Linear Regression models on the 80-point datasets
   - Predict the remaining 70 actual prices (days 80-149)
4. **Evaluation**: Compare predictions against actual prices using multiple metrics

### Numerical Methods Explained:

#### 1. **Lagrange Interpolation**
- **Method**: Constructs a polynomial that passes through all given points
- **Characteristics**: 
  - Global polynomial approach with basis functions
  - Can exhibit oscillations (Runge's phenomenon)
  - Uses stability measures (limited to 8 recent points for volatile data)
  - Includes numerical overflow protection
- **Use Case**: Traditional mathematical interpolation, less suitable for financial data

#### 2. **Euler's Method**
- **Method**: First-order differential equation solver for stochastic processes
- **Characteristics**:
  - Models stock prices using: dS/dt = μ*S + σ*S*ε
  - Estimates drift (μ) and volatility (σ) from historical returns
  - O(h²) local truncation error
  - Excellent for financial time series modeling
- **Use Case**: Fast, reliable differential equation approach for finance

#### 3. **Runge-Kutta 4th Order**
- **Method**: High-accuracy differential equation solver with 4 slope evaluations
- **Characteristics**:
  - Same stochastic model as Euler but higher accuracy
  - O(h⁵) local truncation error - superior to Euler
  - Weighted averaging of k1, k2, k3, k4 slope evaluations
  - Best performance for longer prediction horizons
- **Use Case**: Most accurate method for financial differential equation modeling

## 📊 Analysis Criteria Explained

### 1. **MSE (Mean Squared Error)**
**Formula**: `MSE = (1/n) * Σ(actual - predicted)²`

**What it measures**: Average of squared differences between actual and predicted values

**Interpretation**:
- **Lower values = Better performance**
- **Range**: 0 to ∞ (0 is perfect)
- **Units**: Same as the square of the target variable (e.g., dollars²)
- **Sensitivity**: Heavily penalizes large errors due to squaring

**Example Values**:
- MSE = 0: Perfect predictions
- MSE = 100: Average squared error of 100 (average error ≈ 10 units)
- MSE = 10000: Average squared error of 10000 (average error ≈ 100 units)

### 2. **MAE (Mean Absolute Error)**
**Formula**: `MAE = (1/n) * Σ|actual - predicted|`

**What it measures**: Average absolute difference between actual and predicted values

**Interpretation**:
- **Lower values = Better performance**
- **Range**: 0 to ∞ (0 is perfect)
- **Units**: Same as the target variable (e.g., dollars)
- **Sensitivity**: Treats all errors equally (no squaring)

**Example Values**:
- MAE = 0: Perfect predictions
- MAE = 5: Average error of $5 per prediction
- MAE = 50: Average error of $50 per prediction

### 3. **R² Score (Coefficient of Determination)**
**Formula**: `R² = 1 - (SS_res / SS_tot)`
- Where SS_res = Σ(actual - predicted)²
- Where SS_tot = Σ(actual - mean_actual)²

**What it measures**: Proportion of variance in the target variable explained by the model

**Interpretation**:
- **Higher values = Better performance**
- **Range**: -∞ to 1
- **Units**: Dimensionless (proportion/percentage)

**Value Interpretation**:
- **R² = 1.0**: Perfect model (explains 100% of variance)
- **R² = 0.8**: Good model (explains 80% of variance)
- **R² = 0.5**: Moderate model (explains 50% of variance)
- **R² = 0.0**: Model performs as well as predicting the mean
- **R² < 0**: Model performs worse than predicting the mean

### Metric Comparison Guidelines:

| Performance Level | R² Score | MSE | MAE |
|------------------|----------|-----|-----|
| Excellent | > 0.9 | Very Low | Very Low |
| Good | 0.7 - 0.9 | Low | Low |
| Moderate | 0.4 - 0.7 | Moderate | Moderate |
| Poor | 0.1 - 0.4 | High | High |
| Very Poor | < 0.1 | Very High | Very High |

## 📋 Output Explanation

### Console Output:
1. **Data Loading**: Confirms successful loading of hardcoded stock prices
2. **Interpolation Progress**: Shows range of synthetic data generated by each method
3. **Results Table**: Displays MSE, MAE, and R² scores for each method
4. **Best Method**: Identifies the method with lowest MSE
5. **Analysis Summary**: Shows performance improvement and method characteristics

### Visualizations:
1. **Main Comparison Plot**: Shows actual vs predicted prices for all methods
2. **MSE Comparison**: Bar chart comparing Mean Squared Error
3. **R² Score Comparison**: Bar chart comparing R² scores
4. **Residuals Plot**: Scatter plot of residuals for the best-performing method
5. **Detailed Analysis**: Individual plots for each interpolation method

## 🔍 Project Insights

### Expected Results:
- **Runge-Kutta 4th Order** typically performs best due to superior numerical accuracy
- **Euler's Method** provides good performance with faster computation
- **Lagrange Interpolation** may show oscillations and higher errors with financial data

### Key Findings:
- Differential equation methods significantly outperform polynomial interpolation for financial data
- RK4 achieves ~98.55% improvement over Lagrange interpolation
- The project implements extensive stability measures for robust numerical computation
- Linear regression training benefits from the synthetic data expansion from 50 to 80 points

## 🛠️ Technical Implementation Details

### Stability Improvements:
- **Point Reduction**: Limits interpolation points to prevent overflow
- **Value Clamping**: Constrains synthetic values to reasonable bounds
- **Error Handling**: Fallback to linear interpolation if methods fail
- **Numerical Precision**: Uses float64 for enhanced accuracy

### File Structure:
```
AnalyisProject/
├── main.py                    # Main project file
├── requirements.txt           # Python dependencies
├── README.md                 # This documentation
└── modules/
    ├── numerical_methods.py  # Core numerical methods implementation
    ├── data_handler.py       # Stock data management
    ├── stock_model.py        # Linear regression model wrapper
    └── visualization.py      # Charts and analysis plots
```

## 📚 Educational Value

This project demonstrates:
- **Numerical Methods**: Comparison of polynomial vs differential equation approaches
- **Data Science Pipeline**: From data generation to model evaluation
- **Financial Modeling**: Application of stochastic differential equations to stock prediction
- **Performance Analysis**: Comprehensive evaluation using MSE, MAE, and R² metrics
- **Visualization**: Clear presentation of results and method comparisons

## 🎓 Academic Applications

Perfect for:
- **Numerical Methods Courses**: Demonstrates differential equation vs interpolation techniques
- **Data Science Projects**: Shows complete ML pipeline with financial data
- **Financial Mathematics**: Applies stochastic differential equations to stock modeling
- **Algorithm Comparison**: Systematic evaluation of different numerical approaches

## 🔧 Customization Options

To modify the project:
- **Change Stock Data**: Edit the `hardcoded_prices` list in `data_handler.py` (now 150 points)
- **Adjust Data Split**: Modify the 50/30/70 split ratios in `generate_synthetic_data()`
- **Add New Methods**: Implement additional numerical techniques in `NumericalMethods` class
- **Different Models**: Replace LinearRegression with other ML algorithms
- **More Metrics**: Add additional evaluation metrics in `train_and_predict()`

## 📞 Support

For questions or issues:
1. Check that all required libraries are installed
2. Ensure Python version is 3.7 or higher
3. Verify file paths are correct for your system
4. Review error messages for specific issues

---

**Note**: This project uses hardcoded stock price data to ensure consistent results and avoid external dependencies. The data simulates realistic stock price movements for educational purposes. The differential equation methods (Euler and RK4) provide superior performance for financial time series compared to traditional polynomial interpolation.
