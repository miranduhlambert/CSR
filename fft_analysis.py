import pandas as pd
import numpy as np

#Load the data created by AHK1A_data_preprocessing
taicu_data = pd.read_csv('taicu_data_avg.csv')
#Take the data and store in vectors
time=taicu_data['Adjusted Time'].values
temperature=taicu_data['Temperature Values'].values
n=len(time)
#Find the sampling intervals
sampling_intervals=np.diff(time)
average_sampling_interval=np.mean(sampling_intervals)
sampling_frequency=1/average_sampling_interval
print(f'Sampling Frequency in Cycles/Day is: {sampling_frequency*86400}')
fft_result=np.fft.fft(temperature)
fft_freq=np.fft.fftfreq(n, d=average_sampling_interval)
# Filter only positive frequencies
positive_idxs = np.where(fft_freq >= 0)
fft_freq = fft_freq[positive_idxs]
fft_result = fft_result[positive_idxs]

A_n=np.real(fft_result)
B_n=np.imag(fft_result)

# Create a DataFrame for the FFT results
fft_taicu = pd.DataFrame({
    'Frequency (Cycles/Day)': fft_freq*86400,
    'Amplitude': np.abs(fft_result)*2/n,
    'Phase:': np.arctan2(B_n, A_n) #Using np.angle will not give you the result you want
})
# Save the DataFrame to a text file (CSV format)
fft_taicu.to_csv('fft_taicu  .csv', index=False)