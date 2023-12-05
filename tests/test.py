import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Example data points
x_values = np.array([0, 1, 2, 3, 4, 5])  # Common x-values for both lines
y_values_line1 = np.array([0, 2, 4, 6, 8, 10])  # Y-values for line 1
y_values_line2 = np.array([10, 8, 6, 4, 2, 0])  # Y-values for line 2

# Interpolate the lines
line1 = interp1d(x_values, y_values_line1, kind='linear')
line2 = interp1d(x_values, y_values_line2, kind='linear')

# Find the closest points
x_fine = np.linspace(x_values.min(), x_values.max(), 1000)  # Fine grid of x-values
differences = np.abs(line1(x_fine) - line2(x_fine))
min_index = np.argmin(differences)

# Determine the x-coordinate of the closest point
closest_x = x_fine[min_index]
closest_y = (line1(closest_x) + line2(closest_x)) / 2  # Average y value at the closest x

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values_line1, 'bo-', label='Line 1')
plt.plot(x_values, y_values_line2, 'ro-', label='Line 2')
plt.plot(closest_x, closest_y, 'g*', markersize=15, label='Closest Intersection Point')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Closest Intersection Point Between Two Lines')
plt.legend()
plt.grid(True)
plt.savefig('test_intersect.png')
plt.close()

