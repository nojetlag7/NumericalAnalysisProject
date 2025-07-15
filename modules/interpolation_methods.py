import numpy as np
import math

class InterpolationMethods:
    # This class contains different ways to create synthetic (fake) data using interpolation

    @staticmethod
    def lagrange_interpolation(x_points, y_points, x_target):
        """
        # Lagrange interpolation method
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
        if n > 15:
            indices = np.arange(0, n, 3)
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
        for i in range(len(x_target)):
            total = 0.0
            for j in range(n):
                term = y_points[j]
                for k in range(n):
                    if k != j:
                        denominator = x_points[j] - x_points[k]
                        if abs(denominator) < 1e-10:
                            continue
                        term *= (x_target[i] - x_points[k]) / denominator
                        if abs(term) > 1e10:
                            term = np.sign(term) * 1e10
                total += term
            result[i] = total
        return result

    @staticmethod
    def newton_divided_difference(x_points, y_points, x_target):
        """
        # Newton's divided difference interpolation method
        # This method builds a table of differences to fit a curve through the points.
        # It is good for adding new points without recalculating everything.
        #
        # Args:
        #   x_points: The x values we know
        #   y_points: The y values we know
        #   x_target: The x values we want to estimate
        # Returns:
        #   Interpolated y values for x_target
        """
        n = len(x_points)
        if n > 15:
            indices = np.arange(0, n, 3)
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
        divided_diff = np.zeros((n, n), dtype=np.float64)
        divided_diff[:, 0] = y_points
        for j in range(1, n):
            for i in range(n - j):
                denominator = x_points[i + j] - x_points[i]
                if abs(denominator) < 1e-10:
                    divided_diff[i, j] = 0
                else:
                    divided_diff[i, j] = (divided_diff[i + 1, j - 1] - divided_diff[i, j - 1]) / denominator
        result = np.zeros_like(x_target, dtype=np.float64)
        for i, x in enumerate(x_target):
            value = divided_diff[0, 0]
            for j in range(1, n):
                term = divided_diff[0, j]
                for k in range(j):
                    term *= (x - x_points[k])
                    if abs(term) > 1e10:
                        term = np.sign(term) * 1e10
                        break
                value += term
            result[i] = value
        return result

    @staticmethod
    def newton_forward_difference(x_points, y_points, x_target):
        """
        # Newton's forward difference method (for extrapolation)
        # This method is best when the x values are equally spaced (like days).
        # It uses a table of differences to estimate new values, especially for values outside the known range.
        #
        # Args:
        #   x_points: The x values we know (should be equally spaced)
        #   y_points: The y values we know
        #   x_target: The x values we want to estimate (extrapolate)
        # Returns:
        #   Extrapolated y values for x_target
        """
        n = len(x_points)
        h = x_points[1] - x_points[0]
        if n > 15:
            indices = np.arange(0, n, 3)
            x_points = x_points[indices]
            y_points = y_points[indices]
            n = len(x_points)
            h = x_points[1] - x_points[0]
        forward_diff = np.zeros((n, n), dtype=np.float64)
        forward_diff[:, 0] = y_points
        for j in range(1, n):
            for i in range(n - j):
                forward_diff[i, j] = forward_diff[i + 1, j - 1] - forward_diff[i, j - 1]
        result = np.zeros_like(x_target, dtype=np.float64)
        for i, x in enumerate(x_target):
            u = (x - x_points[0]) / h
            value = forward_diff[0, 0]
            u_term = 1.0
            for j in range(1, min(n, 15)):
                u_term *= (u - (j - 1))
                factorial_j = math.factorial(j)
                term = (u_term * forward_diff[0, j]) / factorial_j
                if abs(term) > 1e10 or np.isinf(term) or np.isnan(term):
                    break
                value += term
            result[i] = value
        return result
