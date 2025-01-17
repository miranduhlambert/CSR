#Imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the range of time in seconds
start_time = 0
end_time =2678400 # 86400 seconds = 24 hours ; 604800 seconds= 1 week;  2,678,288 seconds= 1 month; 31,536,000 seconds= 1 year


# Create the time vector with a step size of 1 second
time_vector = np.arange(start_time, end_time + 1, 1)
N=len(time_vector)
L = end_time - start_time  # Interval lengthcd


#Let's Construct a Synthetic time Series 

#Define the value for fundamental frequency
fundamental_frequency=[2*np.pi/86400, 2*np.pi/86400]

#Choose some constants
a_n= [-0.2, -0.1]

#Choose some more constacts
b_n=[-0.2, -0.1] 

#Choose n
n=[2, 15.451]

# Create synthetic time series
synthetic_signal = np.zeros_like(time_vector, dtype=np.float64)

#Choose a DC for the Signal
A_0=21.3

#Add the DC component into the Signal
synthetic_signal+=A_0
for freq, n, amp_sin, amp_cos in zip(fundamental_frequency, n, a_n, b_n):
    synthetic_signal += amp_cos* np.cos(freq*n*time_vector)-amp_sin *np.sin(freq*n*time_vector) 
# Plot the synthetic signal
# plt.figure(figsize=(14, 7))
# plt.plot(time_vector, synthetic_signal, 'or', label='Synthetic Signal')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.title('Synthetic Signal')
# plt.legend()
# plt.grid(True)
# plt.show()

# Perform FFT on synthetic signal
fft_result_synthetic = np.fft.fft(synthetic_signal)
fft_freq_synthetic = np.fft.fftfreq(len(time_vector), d=1)  # Sampling interval of 1 second

# Filter only positive frequencies
positive_freq_idxs = np.where(fft_freq_synthetic >= 0)
fft_freq_synthetic = fft_freq_synthetic[positive_freq_idxs]
fft_result_synthetic = fft_result_synthetic[positive_freq_idxs]
A_n=np.real(fft_result_synthetic)
B_n=np.imag(fft_result_synthetic)

# Plot the FFT result (magnitude spectrum)
# plt.figure(figsize=(14, 7))
# plt.plot(fft_freq_synthetic*86400, np.abs(fft_result_synthetic)*2/N, 'or')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Frequency (Cycles/Day)')
# plt.ylabel('Amplitude')
# plt.title('FFT of Synthetic Signal')
# plt.grid(True)
# plt.show()

# Create a DataFrame for the FFT results
fft_df = pd.DataFrame({
    'Frequency (Cycles/Day)': fft_freq_synthetic*86400,
    'Amplitude': np.abs(fft_result_synthetic)*2/N,
    'Phase': np.arctan2(B_n,A_n) #Using np.angle will not give you the result you want
})

# Save the DataFrame to a text file (CSV format)
fft_df.to_csv('fft_synthetic_results.csv', index=False)

#Let's Create another Synthetic Time Series in a different representation using angular frequencies and phase shift
#Defining constants
omega=[2*np.pi/5592, 2*np.pi/43200]
amplitudes=[np.sqrt(0.1**2+0.1**2),np.sqrt(0.2**2+.2**2)]

#Portion of Code needed
phi=[np.arctan2(-0.1, -0.1),np.arctan2(-0.2, -0.2)]

#Initializing signal
signal=np.zeros_like(time_vector, dtype=np.float64)

#Adding DC Component
signal+=A_0

#Add the Sinusoidal Components
for freq, amp, phi in zip(omega, amplitudes, phi):
    signal+=amp*np.cos(freq*time_vector+phi)

# Plot the synthetic signal
# plt.figure(figsize=(14, 7))
# plt.plot(time_vector, signal, 'or', label='Synthetic Signal Pt. 2')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.title('Synthetic Signal Pt. 2')
# plt.legend()
# plt.grid(True)
# plt.show()

# Perform FFT on synthetic signal
fft_result = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(len(time_vector), d=1)  # Sampling interval of 1 second

# Filter only positive frequencies
positive_idxs = np.where(fft_freq >= 0)
fft_freq = fft_freq[positive_idxs]
fft_result = fft_result[positive_idxs]
A_n=np.real(fft_result)
B_n=np.imag(fft_result)

# Plot the FFT result (magnitude spectrum)
# plt.figure(figsize=(14, 7))
# plt.plot(fft_freq*86400, np.abs(fft_result)*2/N, 'or')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Frequency (Cycles/Day)')
# plt.ylabel('Amplitude')
# plt.title('FFT of Synthetic Signal')
# plt.grid(True)
# plt.show()

# Create a DataFrame for the FFT results
fft_df_2 = pd.DataFrame({
    'Frequency (Cycles/Day)': fft_freq*86400,
    'Amplitude': np.abs(fft_result)*2/N,
    'Phase:': np.arctan2(B_n, A_n) #Using np.angle will not give you the result you want
})

# Save the DataFrame to a text file (CSV format)
fft_df_2.to_csv('fft_synthetic_results_2.csv', index=False)