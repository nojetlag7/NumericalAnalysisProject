# Numerical Methods Stock Price Prediction Project

## 📊 Project Overview

This project compares three numerical interpolation methods for generating synthetic stock price data and evaluates their effectiveness in training machine learning models for price prediction. The project demonstrates the practical application of numerical methods in financial data analysis.

### 🎯 Project Goals

1. **Compare three interpolation methods:**
   - Lagrange interpolation
   - Newton's divided difference interpolation
   - Newton's forward difference (for extrapolation)

2. **Generate synthetic training data** using only the first 50 real stock prices
3. **Train separate Linear Regression models** for each interpolation method
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
- Load hardcoded stock price data (200 days)
- Generate synthetic data using each interpolation method
- Train Linear Regression models
- Display results table and performance metrics
- Show comprehensive visualizations

## 📈 Project Methodology

### Data Flow:
1. **Original Data**: 200 hardcoded stock prices simulating realistic market movement
2. **Training Phase**: 
   - Use first 50 real prices for interpolation
   - Generate 50 synthetic prices (days 51-100) using each method
   - Combine into 100-point training datasets
3. **Prediction Phase**:
   - Train Linear Regression models on the 100-point datasets
   - Predict the remaining 100 actual prices (days 101-200)
4. **Evaluation**: Compare predictions against actual prices using multiple metrics

### Interpolation Methods Explained:

#### 1. **Lagrange Interpolation**
- **Method**: Constructs a polynomial that passes through all given points
- **Characteristics**: 
  - Global polynomial approach
  - Can exhibit oscillations (Runge's phenomenon)
  - Computationally expensive for large datasets
- **Use Case**: Works well for smooth, well-behaved data

#### 2. **Newton's Divided Difference**
- **Method**: Builds polynomial using divided differences table
- **Characteristics**:
  - Efficient for adding new data points
  - More stable than Lagrange for computation
  - Same polynomial as Lagrange but different computation
- **Use Case**: Preferred when data points are frequently added

#### 3. **Newton's Forward Difference**
- **Method**: Uses forward differences for extrapolation
- **Characteristics**:
  - Designed for equally spaced data points
  - Better suited for extrapolation beyond known range
  - Uses factorial terms in computation
- **Use Case**: Ideal for time series forecasting

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
- **Newton's Divided Difference** typically performs best for smooth interpolation
- **Lagrange Interpolation** may show oscillations with high-degree polynomials
- **Newton's Forward Difference** is designed for extrapolation but may vary in interpolation performance

### Key Findings:
- Polynomial interpolation methods can produce unrealistic values for financial data
- The project implements stability measures (point reduction, value clamping) to handle numerical issues
- Linear regression training benefits from the synthetic data expansion from 50 to 100 points

## 🛠️ Technical Implementation Details

### Stability Improvements:
- **Point Reduction**: Limits interpolation points to prevent overflow
- **Value Clamping**: Constrains synthetic values to reasonable bounds
- **Error Handling**: Fallback to linear interpolation if methods fail
- **Numerical Precision**: Uses float64 for enhanced accuracy

### File Structure:
```
AnalyisProject/
├── main.py              # Main project file
├── requirements.txt     # Python dependencies
└── README.md           # This documentation
```

## 📚 Educational Value

This project demonstrates:
- **Numerical Methods**: Practical implementation of classical interpolation techniques
- **Data Science Pipeline**: From data generation to model evaluation
- **Financial Modeling**: Application of mathematical methods to stock price prediction
- **Performance Analysis**: Comprehensive evaluation using multiple metrics
- **Visualization**: Clear presentation of results and comparisons

## 🎓 Academic Applications

Perfect for:
- **Numerical Methods Courses**: Demonstrates interpolation techniques
- **Data Science Projects**: Shows complete ML pipeline
- **Financial Mathematics**: Applies numerical methods to financial data
- **Algorithm Comparison**: Systematic evaluation of different approaches

## 🔧 Customization Options

To modify the project:
- **Change Stock Data**: Edit the `hardcoded_prices` list in `fetch_stock_data()`
- **Adjust Data Split**: Modify the 50/50/100 split ratios in `generate_synthetic_data()`
- **Add New Methods**: Implement additional interpolation techniques in `InterpolationMethods`
- **Different Models**: Replace LinearRegression with other ML algorithms
- **More Metrics**: Add additional evaluation metrics in `train_and_predict()`

## 📞 Support

For questions or issues:
1. Check that all required libraries are installed
2. Ensure Python version is 3.7 or higher
3. Verify file paths are correct for your system
4. Review error messages for specific issues

---

**Note**: This project uses hardcoded stock price data to ensure consistent results and avoid external dependencies. The data simulates realistic stock price movements for educational purposes.
