#Imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import find_peaks


#Function that checks the Product Flag Sequences for Vectorizing data
def check_product_flag(product_flag):
    # Define the sequences you are looking for
    sequences = {
        'tesu': '00111100000000000100000001000000', #bit 14 product flag; Unknown on Offred; TSU_Y+
        'taicu':'00111100000000001000000001000000',  #bit 15 product flag; GF ACC_ICU temps; TICUN
        'tisu': '00111100000000010000000001000000',  #bit 16 product flag; Most likely GF1 ACC_Feeu; TSU_Y-
        }
    
    #Using the key in the defined sequences, if the porduct flag is in the key return key 
    for key in sequences:
        if product_flag in sequences[key]:
            # If you find a match, exit the loop and return True
            return key
   
    #if no match is found, return None
    return None

#Function for filtering data to Bypass Yaml Header marker
def process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors):
    #Define Variables for Lop Iteration
    marker_found= False
    reference_time_seconds=None

    with open(filename,'r') as file:

        for line in file:

            if marker_found: # snippet to get all the data part the marker, do not have to set marker_found == True because it is defined with  
        
                columns = line.split()

              # Ensure the line has at least 5 columns to avoid index error
                if len(columns) > max(product_flag_index, product_column_index):
                    product_flag = columns[product_flag_index]
                    product_data = float(columns[product_column_index])
                    product_flag_key = check_product_flag(product_flag)

                    if product_flag_key in data_vectors:
                        #Append data to the corresponding key in data_vectors    
                        data_vectors[product_flag_key].append(float(product_data))

                        if reference_time_seconds is None:
                            reference_time_seconds = float(columns[0])

                        time_ms = float(columns[1]) / 1000000  # Convert microseconds to seconds
                        time_seconds = float(columns[0]) - reference_time_seconds
                        time_vectors[product_flag_key].append(time_seconds + time_ms)  # Combine seconds and milliseconds

            elif marker in line:
                marker_found=True


def perform_fft(data_vectors, time_vectors):
    fft_results = {}
    target_freqs= [1/86400, 2/86400, 6/86400, 8/86400, 10/86400, 15/86400, 30/86400, 45/86400]
   
    for product_flag, data in data_vectors.items():
        
        if not data:
            continue
        
        # Calculate sampling frequency using time vectors
        time_vector = np.array(time_vectors[product_flag])
        data=np.array(data)
        sampling_intervals = np.diff(time_vector)
        average_sampling_interval = np.mean(sampling_intervals)
        sampling_frequency= 1/average_sampling_interval
        print(f'Sampling Frequency in Cycles/Day for {product_flag} is : {sampling_frequency*86400}')
    
        # Perform FFT
        fft_result = np.fft.fft(data)
        n=len(data)
        fft_freq = np.fft.fftfreq(n, d=average_sampling_interval)
        
        # Calculate amplitudes and phases
        amplitude = np.abs(fft_result)
        phase = np.angle(fft_result)

        # Extract the DC component (A_0)
        A_0 = fft_result[0].real / n
        print(f"DC Component (A_0) for {product_flag}: {A_0}")
        
        # Print coefficients for closest match to target frequencies
        for target_freq in target_freqs:
            closest_idx = np.argmin(np.abs(fft_freq - target_freq))
            print(f"Product Flag: {product_flag}")
            print(f"Target Frequency: {target_freq} Hz")
            print(f"Closest Frequency: {fft_freq[closest_idx]} Hz")
            print(f"Amplitude: {amplitude[closest_idx]}")
            print(f"Phase: {phase[closest_idx]}")
            print(f"FFT Coefficient: {fft_result[closest_idx]}")
            print("")

        # Save FFT results in Code
        fft_results[product_flag] = (fft_freq, fft_result)

        # Create a DataFrame for the FFT results

        fft_df = pd.DataFrame({
            'Frequency (Cycles/Day)': fft_results[product_flag][0] * 86400,
            'Amplitude': np.abs(fft_results[product_flag][1]),
            'Phase(Radians)': np.angle(fft_results[product_flag][1])
        })
        
        # Save the DataFrame to a text file (CSV format)
        fft_df.to_csv(f'fft_{product_flag}_results.csv', index=False)


        # Plot FFT results (optional)
        plt.figure(figsize=(14, 10))
        plt.subplot(4, 1, 1)
        plt.plot(fft_freq[fft_freq > 0]*86400, np.abs(fft_result[fft_freq > 0]), 'or', label='FFT')  # FFT plot; 5400 seconds is about the time of one revolution
        plt.axvline(x=sampling_frequency*86400, color='r', linestyle='--', label=f'Sampling Frequency: {sampling_frequency*(24*3600)} Hz')
        plt.axvline(x=sampling_frequency*86400/2, color='g', linestyle='--', label=f'Nyquist Frequency: {sampling_frequency*(24*3600/2)} Hz')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (cycles/day)')
        plt.ylabel('Amplitude')
        plt.title(f'Frequency Spectrum for {product_flag}')
        plt.grid(True)
        plt.legend()

        #Plot Frequency Versus Phase
        plt.subplot(4,1,2)
        plt.plot(fft_freq[fft_freq > 0]*86400, phase[fft_freq>0], 'or')
        plt.xscale('log')
        plt.xlabel('Frequency (cycles/day)')
        plt.ylabel('Phase Angle')
        plt.title(f'Frequency Versus Phase {product_flag}')
        plt.grid(True)
        plt.legend()

        # Perform Welch's method for PSD estimation
        plt.subplot(4, 1, 3)
        frequencies, psd = signal.welch(data, fs=sampling_frequency, window='hann', nperseg=len(data))
        plt.semilogy(frequencies*86400, psd, 'or', label='Welch\'s Method')
        plt.axvline(x=sampling_frequency*86400, color='r', linestyle='--', label=f'Sampling Frequency: {sampling_frequency*(24*3600)} Hz')
        plt.axvline(x=sampling_frequency*86400/2, color='g', linestyle='--', label=f'Nyquist Frequency: {sampling_frequency*(24*3600/2)} Hz')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (cycles/day)')
        plt.ylabel('Power Spectral Density')
        plt.title(f'Power Spectral Density (Welch\'s Method) for {product_flag}')
        plt.grid(True)
        plt.legend()

        #Plot Data
        plt.subplot(4,1,4)
        plt.plot(time_vector, data, 'or', label=f'Original Data for {product_flag}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature (Celsius)')
        plt.title(f'Original Data for {product_flag}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        
    return fft_results

#This function takes the values given from the fft_analysis and constructs a sinusoidal model
def construct_sinusoidal_model(fft_results, time_vectors):
    sinusoidal_models= {}
    
    for product_flag, (fft_freq, fft_result) in fft_results.items():
        
        print(f"Processing Product Flag: {product_flag}")
        
        if not fft_result.any():
            print(f"No FFT result for {product_flag}")
            continue

        if product_flag not in time_vectors:
            print(f"No time vector for {product_flag}")
            continue

        time_vector = np.array(time_vectors[product_flag])  # Ensure time_vector is a numpy array
        num_samples = len(time_vector)
        reconstructed_signal = np.zeros(num_samples)

        # Add each frequency component to the reconstructed signal
        for k in range(num_samples):
            amplitude = np.abs(fft_result[k]) / num_samples
            phase = np.angle(fft_result[k])+np.pi
            frequency = fft_freq[k]

            # Calculate the initial phase shift
            initial_phase = phase - 2 * np.pi * frequency * time_vector[0]
            
            # Construct the sinusoidal component
            sinusoidal_component = amplitude * np.cos(2 * np.pi * frequency * time_vector + initial_phase)
            
            # Add the component to the reconstructed signal
            reconstructed_signal += sinusoidal_component

        # Save the reconstructed signal
        sinusoidal_models[product_flag] = abs(reconstructed_signal)

        # Debugging print to show structure of sinusoidal_models
        # print("Debugging: Structure of sinusoidal_models")
        # for key, value in sinusoidal_models.items():
        #     print(f"Product Flag: {key}")
        #     print(f"Reconstructed Signal: {value}")
        #     print(f"Length of Reconstructed Signal: {len(value)}")
        #     print("")

    return sinusoidal_models

def plot_comparison(data_vectors, sinusoidal_models, time_vectors):
    for product_flag in data_vectors.keys():
        
        if product_flag not in sinusoidal_models:
            print(f"No sinusoidal model for {product_flag}")
            continue
        
        original_data = data_vectors[product_flag]
        reconstructed_signal = sinusoidal_models[product_flag]
        time_vector = time_vectors[product_flag]

        # Ensure original_data is converted to numerical values and not lists or other types
        if isinstance(original_data, list):
            original_data = np.array(original_data)  # Convert to numpy array if original_data is a list

        # Apply abs() if original_data contains numerical values
        if isinstance(original_data, (np.ndarray, list)):
            original_data = np.abs(original_data)  # Take absolute value of elements

        plt.figure(figsize=(14, 10))
        plt.plot(time_vector, original_data, label=f'Original data for {product_flag}')
        print(f'plotting original data')
        plt.plot(time_vector, reconstructed_signal, label=f'Reconstructed data for {product_flag}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature (Celsius)')
        plt.title(f'Comparison for {product_flag}')
        plt.legend()  # Add legend
        plt.grid(True)
        plt.show()


        sinusoidal_models[product_flag] = reconstructed_signal

#Define Variables for the Data Bypass

filename=r'C:\data\AHK1A_2024-05-01_C_04.txt' #the r with the prefix denotes the absolute path of the file C:\data\AHK1A 2022_07_01-08_31 C
marker='# End of YAML header'
product_flag_index= 5                        # Adjust this index if the product flag is in a different column
product_column_index= 7                      # Adjust this index to the column where product data is located
time_vector_index= 0                         # Subtract the starting value of the file per every time value to get seconds passed


# Initialize empty lists for storing data
data_vectors = {                            
    'tesu': [],
    'taicu': [],
    'tisu': [],
    # Add more product flags as needed
    }
time_vectors = {  # Initialize empty time vectors for each product flag
    'tesu': [],
    'taicu': [],
    'tisu': [],
}

#Call the function
process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors)

# Skipping Filter data vectors and time vectors for each product flag
# for key in data_vectors:
#     data_vectors[key] = data_vectors[key][::2]  # Skip every other data point
#     time_vectors[key] = time_vectors[key][::2]  # Skip corresponding time points

# Averaging Filter data vectors and time vectors for each product flag
for key in data_vectors:
    averaged_data = []
    averaged_time = []

    for i in range(0, len(data_vectors[key]), 2):
        # Average two consecutive data points
        avg_data = (data_vectors[key][i] + data_vectors[key][i + 1]) / 2.0
        avg_time = (time_vectors[key][i] + time_vectors[key][i + 1]) / 2.0

        averaged_data.append(avg_data)
        averaged_time.append(avg_time)

    # Replace original data vectors and time vectors with averaged data
    data_vectors[key] = averaged_data
    time_vectors[key] = averaged_time

#perform fft on data
fft_results = perform_fft(data_vectors, time_vectors)
print(fft_results['taicu'])

#construct sinusoidal model
sinusoidal_models=construct_sinusoidal_model(fft_results, time_vectors)

#plot sinusoidal model and data vectors form comparison
plot_comparison(data_vectors, sinusoidal_models, time_vectors)


