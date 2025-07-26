import numpy as np
import math

class NumericalMethods:
    # This class contains different numerical methods to create synthetic stock price data 
    # We use both traditional interpolation and modern differential equation methods for financial modeling

    @staticmethod
    def lagrange_interpolation(x_points, y_points, x_target):
        """
        # Lagrange interpolation method (traditional polynomial approach)
        # This method finds a curve that passes through all the known points.
        # It then uses that curve to estimate (interpolate) new values.
        # We add checks to avoid very large or small numbers (overflow protection).
        #
        # Args:
        #   x_points: The x values we know (e.g., days)
        #   y_points: The y values we know (e.g., prices)
        #   x_target: The x values we want to estimate (interpolate)
        # Returns:
        #   Interpolated y values for x_target
        """
        n = len(x_points)
        result = np.zeros_like(x_target, dtype=np.float64)
        
        # For volatile data, use fewer points and focus on recent trend
        if n > 8:
            # Use last 8 points for better stability with volatile data
            x_points = x_points[-8:]
            y_points = y_points[-8:]
            n = len(x_points)
        
        # Calculate trend for fallback
        if n >= 2:
            recent_slope = (y_points[-1] - y_points[-2]) / (x_points[-1] - x_points[-2])
        else:
            recent_slope = 0
        
        for i in range(len(x_target)):
            total = 0.0
            unstable = False
            
            for j in range(n):
                term = y_points[j]
                denominator_product = 1.0
                numerator_product = 1.0
                
                for k in range(n):
                    if k != j:
                        denominator = x_points[j] - x_points[k]
                        numerator = x_target[i] - x_points[k]
                        
                        if abs(denominator) < 1e-10:
                            unstable = True
                            break
                        
                        denominator_product *= denominator
                        numerator_product *= numerator
                
                if unstable:
                    break
                
                if abs(denominator_product) < 1e-12:
                    unstable = True
                    break
                
                lagrange_basis = numerator_product / denominator_product
                term *= lagrange_basis
                
                # Check for numerical instability
                if abs(term) > 1e6 or np.isinf(term) or np.isnan(term):
                    unstable = True
                    break
                
                total += term
            
            # Use linear extrapolation if unstable or result is unreasonable
            if unstable or abs(total) > 1e6 or np.isinf(total) or np.isnan(total):
                # Fall back to linear extrapolation from last point
                result[i] = y_points[-1] + recent_slope * (x_target[i] - x_points[-1])
            else:
                # Check if result is reasonable (within 5x of last price)
                if abs(total - y_points[-1]) > 5 * abs(y_points[-1]):
                    result[i] = y_points[-1] + recent_slope * (x_target[i] - x_points[-1])
                else:
                    result[i] = total
        
        return result

    @staticmethod
    def euler_method(x_points, y_points, x_target):
        """
        # Euler's method for stock price modeling (differential equation approach)
        # This method models stock price changes as a differential equation:
        # dS/dt = μ*S + σ*S*ε, where μ is drift, σ is volatility, ε is noise
        # This is much more realistic for financial time series than polynomial fitting.
        #
        # Args:
        #   x_points: The x values we know (e.g., days)
        #   y_points: The y values we know (e.g., prices)
        #   x_target: The x values we want to estimate
        # Returns:
        #   Estimated y values using Euler's method
        """
        if len(x_points) < 5:
            # Not enough data, fall back to linear extrapolation
            if len(x_points) >= 2:
                slope = (y_points[-1] - y_points[-2]) / (x_points[-1] - x_points[-2])
                return np.array([y_points[-1] + slope * (x - x_points[-1]) for x in x_target])
            else:
                return np.full_like(x_target, y_points[-1])
        
        # Calculate parameters from recent price history
        # Use last 10 points to estimate drift and volatility
        recent_points = min(10, len(x_points))
        recent_x = x_points[-recent_points:]
        recent_y = y_points[-recent_points:]
        
        # Calculate returns (percentage changes)
        returns = np.diff(recent_y) / recent_y[:-1]
        
        # Estimate drift (average return) and volatility (standard deviation of returns)
        drift = np.mean(returns) if len(returns) > 0 else 0.0
        volatility = np.std(returns) if len(returns) > 1 else 0.01  # Default small volatility
        
        # Limit extreme values for stability
        drift = np.clip(drift, -0.1, 0.1)  # Max 10% daily change
        volatility = np.clip(volatility, 0.001, 0.05)  # 0.1% to 5% daily volatility
        
        # Starting values
        current_price = float(y_points[-1])
        current_time = float(x_points[-1])
        dt = 1.0  # Time step (1 day)
        
        result = np.zeros_like(x_target, dtype=np.float64)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        for i, target_time in enumerate(x_target):
            steps = int(target_time - current_time)
            price = current_price
            
            for step in range(steps):
                # Euler's method: S(t+dt) = S(t) + dt * f(S(t), t)
                # Where f(S,t) = drift*S + volatility*S*random_noise
                
                # Generate controlled random noise (not too extreme)
                noise = np.random.normal(0, 1)
                noise = np.clip(noise, -2, 2)  # Limit to ±2 standard deviations
                
                # Calculate the derivative (rate of change)
                drift_component = drift * price
                volatility_component = volatility * price * noise
                
                # Euler step
                price_change = dt * (drift_component + volatility_component)
                
                # Apply the change
                price += price_change
                
                # Ensure price stays positive and reasonable
                price = max(price, current_price * 0.5)  # Don't drop below 50% of starting price
                price = min(price, current_price * 2.0)   # Don't rise above 200% of starting price
            
            result[i] = price
        
        return result

    @staticmethod
    def runge_kutta_4th_order(x_points, y_points, x_target):
        """
        # Runge-Kutta 4th order method for stock price modeling (high-accuracy differential equation approach)
        # This method is much more accurate than Euler's method for solving differential equations.
        # It uses 4 slope evaluations per step to get better accuracy.
        # Models stock price as: dS/dt = μ*S + σ*S*ε (same as Euler for fair comparison)
        #
        # Args:
        #   x_points: The x values we know (e.g., days)
        #   y_points: The y values we know (e.g., prices)
        #   x_target: The x values we want to estimate
        # Returns:
        #   Estimated y values using RK4 method
        """
        if len(x_points) < 5:
            # Not enough data, fall back to linear extrapolation
            if len(x_points) >= 2:
                slope = (y_points[-1] - y_points[-2]) / (x_points[-1] - x_points[-2])
                return np.array([y_points[-1] + slope * (x - x_points[-1]) for x in x_target])
            else:
                return np.full_like(x_target, y_points[-1])
        
        # Calculate parameters from recent price history (same as Euler method)
        recent_points = min(10, len(x_points))  # Use same number of points as Euler
        recent_x = x_points[-recent_points:]
        recent_y = y_points[-recent_points:]
        
        # Calculate returns (percentage changes)
        returns = np.diff(recent_y) / recent_y[:-1]
        
        # Estimate drift (average return) and volatility (standard deviation of returns)
        drift = np.mean(returns) if len(returns) > 0 else 0.0
        volatility = np.std(returns) if len(returns) > 1 else 0.01  # Default small volatility
        
        # Limit extreme values for stability (same constraints as Euler)
        drift = np.clip(drift, -0.1, 0.1)  # Max 10% daily change
        volatility = np.clip(volatility, 0.001, 0.05)  # 0.1% to 5% daily volatility
        
        # Starting values
        current_price = float(y_points[-1])
        current_time = float(x_points[-1])
        dt = 1.0  # Time step (1 day)
        
        # Pre-generate random numbers for fair comparison
        total_steps = sum(int(target_time - current_time) for target_time in x_target)
        np.random.seed(42)  # Set seed before generating
        random_noises = []
        for i in range(total_steps):
            noise = np.random.normal(0, 1)
            noise = np.clip(noise, -2, 2)  # Limit to ±2 standard deviations
            random_noises.append(noise)
        
        def stock_price_derivative(price, step_index):
            """
            Define the differential equation for stock price evolution
            dS/dt = drift*S + volatility*S*noise (same model as Euler)
            """
            # Use the same random noise for all k evaluations in RK4 step
            if step_index < len(random_noises):
                noise = random_noises[step_index]
            else:
                noise = 0.0  # Fallback
            
            # Components of the differential equation (same as Euler)
            drift_component = drift * price
            volatility_component = volatility * price * noise
            
            return drift_component + volatility_component
        
        result = np.zeros_like(x_target, dtype=np.float64)
        step_counter = 0
        
        for i, target_time in enumerate(x_target):
            steps = int(target_time - current_time)
            price = current_price
            
            for step in range(steps):
                # Runge-Kutta 4th order method
                # Use the same noise for all k evaluations in this step
                current_step_index = step_counter
                
                # k1 = dt * f(t, y)
                k1 = dt * stock_price_derivative(price, current_step_index)
                
                # k2 = dt * f(t + dt/2, y + k1/2)
                k2 = dt * stock_price_derivative(price + k1/2, current_step_index)
                
                # k3 = dt * f(t + dt/2, y + k2/2)
                k3 = dt * stock_price_derivative(price + k2/2, current_step_index)
                
                # k4 = dt * f(t + dt, y + k3)
                k4 = dt * stock_price_derivative(price + k3, current_step_index)
                
                # Combine the slopes: y(t+dt) = y(t) + (k1 + 2*k2 + 2*k3 + k4)/6
                price_change = (k1 + 2*k2 + 2*k3 + k4) / 6
                price += price_change
                
                # Ensure price stays positive and reasonable (same bounds as Euler)
                price = max(price, current_price * 0.5)  # Don't drop below 50% of starting price
                price = min(price, current_price * 2.0)   # Don't rise above 200% of starting price
                
                step_counter += 1
            
            result[i] = price
        
        return result
