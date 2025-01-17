import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from math import sqrt  # Import sqrt from the math module

# Function to fit the slope in log-log space
def fit_data_log_log(freq, amp):
    # Filter out zero frequencies and amplitudes to avoid log(0)
    valid_indices = (freq > 0) & (amp > 0)
    log_freq = np.log10(freq[valid_indices])
    log_amp = np.log10(amp[valid_indices])
    
    # Perform linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = linregress(log_freq, log_amp)
    print("Log-log slope: %f    intercept: %f" % (slope, intercept))
    return slope, intercept, std_err

# List of data types you want to plot
data_types = ['tesu', 'taicu', 'tisu']

# Loop through each data type to load and plot the data
for data_type in data_types:
    # Load the FFT-processed data for the specific data type
    fft_data = dd.read_csv(f'fft_{data_type}_result.txt', sep='\s+')

    # Compute the Dask DataFrames into pandas DataFrames for processing
    fft_data = fft_data.compute()

    # Extract data types individually
    fft_frequency = fft_data['Frequency (Cycles/Day)'].values
    fft_amplitude = fft_data['Amplitude'].values

    # Call the fit_data_log_log function to get slope and intercept
    slope, intercept, std_err = fit_data_log_log(fft_frequency, fft_amplitude)

    # Generate the fitted line values in log space
    fitted_line_log = slope * np.log10(fft_frequency) + intercept
    fitted_line = 10 ** fitted_line_log  # Convert back to original scale

# Calculate upper and lower bound lines using standard error
# Calculate upper and lower bound lines using standard error with 2 * sqrt(N)
    N = len(fft_frequency)  # Number of data points
    upper_bound_log = fitted_line_log + 3 * std_err * sqrt(N)
    lower_bound_log = fitted_line_log - 3 * std_err * sqrt(N)
    upper_bound = 10 ** upper_bound_log  # Convert back to original scale
    lower_bound = 10 ** lower_bound_log  # Convert back to original scale


# Calculate the 1/f and 1/f^2 lines
    m_f= 1 / fft_frequency
    line_1_over_f = m_f* fft_frequency
    m_ff=1 / (fft_frequency ** 2)
    line_1_over_f_squared = m_ff *fft_frequency

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    plt.loglog(fft_frequency, fft_amplitude, 'o', label=f'FFT Amplitude ({data_type})', markersize=4)
    
    # Plot the fitted line
    plt.loglog(fft_frequency, fitted_line, 'r-', label='Fitted Line')
    
    # Plot the upper and lower bound lines
    plt.loglog(fft_frequency, upper_bound, 'r--', label='Upper Bound')
    plt.loglog(fft_frequency, lower_bound, 'r--', label='Lower Bound')

    # Plot the 1/f line
    plt.loglog(fft_frequency, line_1_over_f, 'g--', label='m = 1/f')

    # Plot the 1/f^2 line
    plt.loglog(fft_frequency, line_1_over_f_squared, 'b--', label='m = 1/f^2')

    plt.xlabel('Frequency (Cycles/Day)')
    plt.ylabel('Amplitude (Temperature)')
    plt.title(f'Log-Log Plot for {data_type}')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()