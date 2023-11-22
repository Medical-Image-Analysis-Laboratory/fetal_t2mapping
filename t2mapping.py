import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate some synthetic data
echo_times = np.array([10, 30, 50, 70, 90])
signal_intensity = np.array([100, 70, 50, 30, 20])

# Define the exponential decay function
def exponential_decay(t, A, T2, C):
    return A * np.exp(-t / T2) + C

# Use curve_fit to fit the data
popt, pcov = curve_fit(exponential_decay, echo_times, signal_intensity)

# Extract fitted parameters
A_fit, T2_fit, C_fit = popt

# Create a time array for plotting the fitted curve
t_fit = np.linspace(min(echo_times), max(echo_times), 100)

# Plot the original data and the fitted curve
plt.scatter(echo_times, signal_intensity, label='Original Data')
plt.plot(t_fit, exponential_decay(t_fit, *popt), 'r-', label='Fit: A=%5.3f, T2=%5.3f, C=%5.3f' % tuple(popt))
plt.xlabel('Echo Time (ms)')
plt.ylabel('Signal Intensity')
plt.legend()
plt.show()