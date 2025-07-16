# Numerical Methods Gap-Filling Analysis Project

## 📊 Project Overview

This project demonstrates a realistic approach to filling gaps in stock price data using three numerical interpolation methods. Unlike traditional synthetic data generation, this project simulates real-world scenarios where some data points are missing and need to be recovered using interpolation techniques combined with machine learning prediction.

### 🎯 Project Goals

1. **Compare three interpolation methods for gap filling:**
   - Lagrange interpolation
   - Newton's divided difference interpolation
   - Cubic Spline interpolation

2. **Realistic gap-filling scenario** using 200 stock prices with 100 missing values
3. **Hybrid approach**: Use interpolation for 50 gaps, ML models for remaining 50 gaps
4. **Evaluate and compare** the effectiveness of each method in a practical context
5. **Visualize results** with comprehensive gap-filling analysis charts

## 🔧 Required Libraries

Before running the project, ensure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### Library Details:
- **NumPy (>=1.21.0)**: Numerical computing and array operations
- **Pandas (>=1.3.0)**: Data manipulation and analysis
- **Matplotlib (>=3.4.0)**: Data visualization and plotting
- **Scikit-learn (>=1.0.0)**: Machine learning algorithms and metrics
- **SciPy (>=1.7.0)**: Scientific computing library for cubic spline interpolation

## 🚀 How to Run the Project

1. **Clone or download** the project files to your local machine
2. **Navigate** to the project directory:
   ```bash
   cd "path/to/NumericalAnalysisProject"
   ```
3. **Run the main script**:
   ```bash
   python main.py
   ```

The program will automatically:
- Load 200 hardcoded stock prices
- Create gaps by removing 100 random values
- Use interpolation methods to fill 50 gaps
- Train ML models to predict remaining 50 gaps
- Display comprehensive results and performance metrics
- Show gap-filling analysis visualizations

## 📈 Project Methodology

### Data Flow:
1. **Original Data**: 200 hardcoded stock prices simulating realistic market movement
2. **Gap Creation**: Randomly remove 100 values (50% of data) to simulate missing data
3. **Gap Filling Phase**: 
   - Use interpolation methods to fill 50 of the missing values
   - Create training dataset with 150 points (100 known + 50 interpolated)
4. **Prediction Phase**:
   - Train Linear Regression models on the 150-point datasets
   - Predict the remaining 50 missing values
5. **Evaluation**: Compare predictions against actual values using multiple metrics

### Interpolation Methods Explained:

#### 1. **Lagrange Interpolation**
- **Method**: Constructs a polynomial that passes through all given points
- **Characteristics**: 
  - Global polynomial approach
  - Can exhibit oscillations with high-degree polynomials
  - Uses equally spaced points for numerical stability
- **Use Case**: Works well for smooth, well-behaved data within known ranges

#### 2. **Newton's Divided Difference**
- **Method**: Builds polynomial using divided differences table
- **Characteristics**:
  - Efficient for adding new data points
  - More stable than Lagrange for computation
  - Uses equally spaced points for optimal performance
- **Use Case**: Preferred when filling gaps in existing datasets

#### 3. **Cubic Spline Interpolation**
- **Method**: Uses piecewise cubic polynomials with smooth transitions
- **Characteristics**:
  - Designed specifically for interpolation (gap-filling)
  - Smooth, continuous curves without oscillations
  - Minimizes curvature for natural-looking interpolation
- **Use Case**: Industry standard for filling gaps in continuous data series

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
1. **Data Loading**: Confirms successful loading of 200 hardcoded stock prices
2. **Gap Analysis**: Shows breakdown of known vs missing data points
3. **Interpolation Progress**: Displays interpolation MSE and value ranges for each method
4. **Training Info**: Shows training dataset size (150 points) for each method
5. **Results Table**: Displays MSE, MAE, and R² scores for model predictions
6. **Best Method**: Identifies the method with lowest prediction MSE
7. **Analysis Summary**: Shows performance improvement and method characteristics

### Visualizations:
1. **Gap Filling Comparison**: Shows original data, known points, interpolated values, and predictions
2. **MSE Comparison**: Bar chart comparing Mean Squared Error for model predictions
3. **R² Score Comparison**: Bar chart comparing R² scores for model predictions
4. **Error Analysis**: Scatter plot comparing interpolation vs prediction errors
5. **Method Performance**: Individual analysis for each interpolation method

## 🔍 Project Insights

### Expected Results:
- **Cubic Spline** typically provides the smoothest interpolation for gap-filling
- **Newton's Divided Difference** provides good balance of accuracy and stability
- **Lagrange Interpolation** may show oscillations but can be very accurate for smooth data

### Key Findings:
- Gap-filling with interpolation is more realistic than synthetic data generation
- The hybrid approach (interpolation + ML) provides robust missing value recovery
- Equally spaced point selection improves numerical stability
- Training on 150 points (100 known + 50 interpolated) provides sufficient data for good predictions

### Real-World Applications:
- **Financial Data**: Filling gaps in stock price, trading volume, or economic indicators
- **Sensor Data**: Recovering missing readings from IoT devices or monitoring systems
- **Medical Data**: Interpolating missing patient measurements or test results
- **Scientific Data**: Filling gaps in experimental or observational datasets

## 🛠️ Technical Implementation Details

### Stability Improvements:
- **Point Reduction**: Limits interpolation points to prevent overflow
- **Value Clamping**: Constrains synthetic values to reasonable bounds
- **Error Handling**: Fallback to linear interpolation if methods fail
- **Numerical Precision**: Uses float64 for enhanced accuracy

### File Structure:
```
NumericalAnalysisProject/
├── main.py                      # Main project file
├── requirements.txt             # Python dependencies
├── README.md                   # This documentation
└── modules/
    ├── __init__.py             # Package initialization
    ├── data_handler.py         # Data loading and management
    ├── interpolation_methods.py # Three interpolation methods
    ├── model.py                # Linear regression model
    └── visualization.py        # Plotting and visualization
```

## 📚 Educational Value

This project demonstrates:
- **Numerical Methods**: Practical implementation of classical interpolation techniques
- **Gap-Filling Analysis**: Realistic approach to missing data problems
- **Hybrid Methodology**: Combining interpolation with machine learning
- **Performance Analysis**: Comprehensive evaluation using multiple metrics
- **Equal Spacing**: Importance of data point selection for numerical stability

## 🎓 Academic Applications

Perfect for:
- **Numerical Methods Courses**: Demonstrates interpolation techniques in practical context
- **Data Science Projects**: Shows realistic gap-filling methodology
- **Financial Mathematics**: Applies numerical methods to financial data analysis
- **Algorithm Comparison**: Systematic evaluation of different interpolation approaches
- **Missing Data Analysis**: Practical approach to handling incomplete datasets

## 🔧 Customization Options

To modify the project:
- **Change Stock Data**: Edit the `hardcoded_prices` list in `data_handler.py`
- **Adjust Gap Ratio**: Modify the 100 missing values or 50/50 split in `main.py`
- **Add New Methods**: Implement additional interpolation techniques in `InterpolationMethods`
- **Different Models**: Replace LinearRegression with other ML algorithms in `model.py`
- **More Metrics**: Add additional evaluation metrics in the `evaluate()` method

## 📞 Support

For questions or issues:
1. Check that all required libraries are installed
2. Ensure Python version is 3.7 or higher
3. Verify file paths are correct for your system
4. Review error messages for specific issues

---

**Note**: This project uses hardcoded stock price data to ensure consistent results and avoid external dependencies. The data simulates realistic stock price movements for educational purposes. The gap-filling approach provides a practical demonstration of how interpolation methods can be used to recover missing data in real-world scenarios.

