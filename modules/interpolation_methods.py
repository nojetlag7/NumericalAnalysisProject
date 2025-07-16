import numpy as np
import math
from scipy.interpolate import CubicSpline

class InterpolationMethods:
    # This class contains three different interpolation methods for gap filling
    # Each method uses equally spaced known data points for numerical stability
    # Used to fill 50 missing values in stock price data

    @staticmethod
    def lagrange_interpolation(x_points, y_points, x_target):
        """
        # Lagrange interpolation method for gap filling
        # Constructs a polynomial that passes through all known points
        # Uses equally spaced points for numerical stability
        # Best for smooth, well-behaved data without oscillations
        """
        n = len(x_points)
        
        # Reduce points if too many, ensuring equal spacing
        if n > 12:
            # Calculate step size to get equally spaced points
            step = max(1, n // 12)
            indices = np.arange(0, n, step)[:12]
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
        
        # Verify equal spacing
        if n > 1:
            steps = np.diff(x_points)
            avg_step = np.mean(steps)
            print(f"    Lagrange: Using {n} points with average step size: {avg_step:.2f}")
        
        result = np.zeros_like(x_target, dtype=np.float64)
        
        for i in range(len(x_target)):
            total = 0.0
            x_val = x_target[i]
            
            for j in range(n):
                # Calculate Lagrange basis polynomial L_j(x)
                basis = 1.0
                
                for k in range(n):
                    if k != j:
                        denominator = x_points[j] - x_points[k]
                        if abs(denominator) < 1e-12:
                            basis = 0.0
                            break
                        
                        numerator = x_val - x_points[k]
                        basis *= numerator / denominator
                        
                        # Prevent overflow
                        if abs(basis) > 1e8:
                            basis = np.sign(basis) * 1e8
                            break
                
                # Add contribution of this basis function
                contribution = y_points[j] * basis
                
                # Check for numerical issues
                if not (np.isfinite(contribution) and abs(contribution) < 1e10):
                    contribution = 0.0
                    
                total += contribution
            
            result[i] = total
        
        return result

    @staticmethod
    def newton_divided_difference(x_points, y_points, x_target):
        """
        # Newton's divided difference interpolation method for gap filling
        # Builds a table of differences to fit a curve through the points
        # Uses equally spaced points for numerical stability
        # Efficient for adding new points without recalculating everything
        """
        n = len(x_points)
        
        # Reduce points if too many, ensuring equal spacing
        # Use a slightly different starting point than Lagrange for variety
        if n > 12:
            # Calculate step size to get equally spaced points
            step = max(1, n // 12)
            # Start from index 1 instead of 0 for different selection
            start_idx = min(1, n - 1)
            indices = np.arange(start_idx, n, step)[:12]
            if len(indices) < 12 and step > 1:
                # If we don't have enough points, adjust
                indices = np.arange(0, n, step)[:12]
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
        
        # Verify equal spacing
        if n > 1:
            steps = np.diff(x_points)
            avg_step = np.mean(steps)
            print(f"    Newton DD: Using {n} points with average step size: {avg_step:.2f}")
        
        # Build divided difference table
        divided_diff = np.zeros((n, n), dtype=np.float64)
        divided_diff[:, 0] = y_points
        
        for j in range(1, n):
            for i in range(n - j):
                denominator = x_points[i + j] - x_points[i]
                if abs(denominator) < 1e-12:
                    divided_diff[i, j] = 0
                else:
                    divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / denominator
        
        result = np.zeros_like(x_target, dtype=np.float64)
        
        for i, x in enumerate(x_target):
            value = divided_diff[0, 0]
            
            for j in range(1, n):
                term = divided_diff[0, j]
                
                # Calculate the product term
                for k in range(j):
                    term *= (x - x_points[k])
                    
                    # Prevent overflow
                    if abs(term) > 1e8:
                        term = np.sign(term) * 1e8
                        break
                
                # Check for numerical issues
                if np.isfinite(term) and abs(term) < 1e10:
                    value += term
                else:
                    break
            
            result[i] = value
        
        return result

    @staticmethod
    def cubic_spline_interpolation(x_points, y_points, x_target):
        """
        # Cubic Spline interpolation method for gap filling
        # Uses piecewise cubic polynomials with smooth transitions
        # Designed specifically for interpolation (gap-filling) applications
        # Minimizes oscillations and provides smooth, continuous curves
        """
        n = len(x_points)
        
        # Reduce points if too many, ensuring equal spacing
        # Use a different selection strategy for variety
        if n > 15:
            # Calculate step size to get equally spaced points
            step = max(1, n // 15)
            # Start from index 1 for different selection pattern
            start_idx = min(1, n - 1)
            indices = np.arange(start_idx, n, step)[:15]
            if len(indices) < 15 and step > 1:
                # If we don't have enough points, adjust
                indices = np.arange(0, n, step)[:15]
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
        
        # Verify spacing and report
        if n > 1:
            steps = np.diff(x_points)
            avg_step = np.mean(steps)
            step_std = np.std(steps)
            print(f"    Cubic Spline: Using {n} points with average step: {avg_step:.2f} (std: {step_std:.3f})")
        
        # Need at least 2 points for cubic spline
        if n < 2:
            return np.full_like(x_target, y_points[0] if len(y_points) > 0 else 0.0)
        
        # For very few points, use linear interpolation
        if n < 4:
            result = np.interp(x_target, x_points, y_points)
            return result
        
        try:
            # Create cubic spline interpolator
            # Use 'natural' boundary conditions (second derivative = 0 at endpoints)
            cs = CubicSpline(x_points, y_points, bc_type='natural')
            
            # Evaluate spline at target points
            result = cs(x_target)
            
            # Clamp results to reasonable bounds to prevent extreme values
            y_min, y_max = np.min(y_points), np.max(y_points)
            y_range = y_max - y_min
            lower_bound = y_min - 0.5 * y_range
            upper_bound = y_max + 0.5 * y_range
            
            result = np.clip(result, lower_bound, upper_bound)
            
            return result
            
        except Exception as e:
            print(f"    Cubic Spline failed: {e}, falling back to linear interpolation")
            # Fallback to linear interpolation
            result = np.interp(x_target, x_points, y_points)
            return result
