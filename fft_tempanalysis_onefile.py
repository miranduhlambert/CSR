#Imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle

#Function for filtering data to Bypass Yaml Header marker
def process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors):

    marker_found= False
    reference_time_seconds=None

    with open(filename,'r') as file:

        for line in file:

            if marker_found: # snippet to get all the data part the marker, do not have to set marker_found == True because it is defined with  
        
                columns = line.split()

                # Ensure the line has at least 5 columns to avoid index errors
                if len(columns) > max(product_flag_index, product_column_index):
        
                    product_flag = columns[product_flag_index]  # Extract the fifth column
                    product_data= columns[product_column_index]
                   
                    #Call check_product_flag to get the product flag key
                    product_flag_key=check_product_flag(product_flag) 

                    # if product flag key is found
                    if product_flag_key in data_vectors:    
                            
                                          
                            #Append data to the corresponding key in data_vectors
                            data_vectors[product_flag_key].append(float(product_data))

                            #Extract the reference time from the first line of each file
                            if reference_time_seconds is None:
                                reference_time_seconds= float(columns[0])
                            time_ms = float(columns[1]) / 1000000  # Convert milliseconds to seconds
                            time_seconds = float(columns[0]) - reference_time_seconds  # Subtract the reference time point
                            time_vectors[product_flag_key].append(time_seconds + time_ms)  # Combine seconds and milliseconds

            elif marker in line:
                marker_found=True
                # print(f"Marker found: {marker}")
    # Sort the time vectors and corresponding data
    sorted_time_vectors, sorted_data_vectors = sort_time_vectors_and_data(time_vectors, data_vectors)

    return sorted_time_vectors, sorted_data_vectors

def sort_time_vectors_and_data(time_vectors, data_vectors):
    sorted_time_vectors = {}
    sorted_data_vectors = {}

    for product_flag in time_vectors:
        time_vector = np.array(time_vectors[product_flag])
        data_vector = np.array(data_vectors[product_flag])

        # Sort time vector and data vector
        sorted_indices = np.argsort(time_vector)
        sorted_time_vector = time_vector[sorted_indices]
        sorted_data_vector = data_vector[sorted_indices]

        sorted_time_vectors[product_flag] = sorted_time_vector.tolist()
        sorted_data_vectors[product_flag] = sorted_data_vector.tolist()

        # Check if time vector is evenly spaced
        sampling_intervals = np.diff(sorted_time_vector)
        average_interval = np.mean(sampling_intervals)
        is_evenly_spaced = np.allclose(sampling_intervals, average_interval, rtol=1e-5, atol=1e-8)


        print(f"Product Flag: {product_flag}")
        print(f"Is evenly spaced: {is_evenly_spaced}")
        if not is_evenly_spaced:
            print(f"First 10 sampling intervals: {sampling_intervals[:10]}")
            print(f"Last 10 sampling intervals: {sampling_intervals[-10:]}")
            print(f"Total number of intervals: {len(sampling_intervals)}")

    return sorted_time_vectors, sorted_data_vectors

def check_product_flag(product_flag):

    # Define the sequences you are looking for
    sequences = {
        'tesu': '00111100000000000100000001000000', #bit 14 product flag, 
        'taicu':'00111100000000001000000001000000',  #bit 15 product flag; GF1 ACC Hk N ICU temps
        'tisu': '00111100000000010000000001000000',  #bit 16 product flag
        # Add more sequences as needed
    }

    for key, sequence in sequences.items():
        # If you find a match, exit the loop and return True
        if product_flag == sequence:
            return key  

    return None  # If no match is found, return None'


def plot_data_vectors(data_vectors,time_vectors):
    for product_flag, data in data_vectors.items():
        if not data:
            print(f"No data to plot for {product_flag}")
            continue
        #Plot the data
        time_vector = time_vectors[product_flag]
        plt.figure()
        plt.plot(time_vector, data, label=product_flag)
        plt.xlabel('Time of Day (seconds)')
        plt.ylabel('Temperature (Celsius)')
        plt.title(f'Product Data Analysis for Product Flag: {product_flag}')
        plt.legend()
        plt.show()

def perform_lomb_scargle(data_vectors, time_vectors):
    for product_flag, data in data_vectors.items():
        if not data:
            continue

        time_vector = np.array(time_vectors[product_flag])
        data = np.array(data)

        # Normalize time vector to start from zero
        time_vector_normalized = time_vector - time_vector[0]

        # Compute Lomb-Scargle periodogram
        ls = LombScargle(time_vector_normalized, data)
        frequency, power = ls.autopower()

        # Convert frequency to cycles per day or any desired unit
        frequency_cpd = frequency * 86400  # Example: Convert to cycles per day

        # Calculate amplitude as square root of power (assuming normalized data)
        amplitude = np.sqrt(power)

        # Plot amplitude versus frequency
        plt.figure(figsize=(10, 6))
        plt.plot(frequency_cpd, amplitude)
        plt.xlabel('Frequency (cycles/day)')
        plt.ylabel('Amplitude')
        plt.title(f'Amplitude versus Frequency (Lomb-Scargle) for {product_flag}')
        plt.grid(True)
        plt.show()

        # Find peaks in the periodogram
        peak_indices = np.argsort(amplitude)[-5:]  # Get indices of the top 5 peaks
        top_frequencies = frequency_cpd[peak_indices]
        top_amplitudes = amplitude[peak_indices]

        print(f"Top 5 frequencies: {top_frequencies}")
        print(f"Corresponding amplitudes: {top_amplitudes}")


def perform_fft(data_vectors, time_vectors):
    fft_results = {}
    
    for product_flag, data in data_vectors.items():
        
        if not data:
            continue
        
        # Calculate sampling frequency using time vectors
        time_vector = np.array(time_vectors[product_flag])
        data=np.array(data)
        sampling_intervals = np.diff(time_vector)
        average_sampling_interval = np.mean(sampling_intervals)
        sampling_frequency = 1 / average_sampling_interval
        
        # Perform FFT
        fft_result = np.fft.fft(data)
        fft_freq = np.fft.fftfreq(len(data), d=average_sampling_interval)

        # Save FFT results
        fft_results[product_flag] = (fft_freq, fft_result)
        
        # Normalize time vector to start from zero
        time_vector_normalized = time_vector - time_vector[0]

        # Compute Lomb-Scargle periodogram
        ls = LombScargle(time_vector_normalized, data, normalization='psd', fit_mean=True)
        frequency, power = ls.autopower()

        # Convert frequency to cycles per day
        frequency_cpd = frequency * 86400

        # Find peaks in the FFT magnitude spectrum
        positive_freqs = fft_freq > 0
        fft_magnitude = np.abs(fft_result)[positive_freqs]
        fft_peaks, _ = find_peaks(fft_magnitude)
        fft_peak_freqs = fft_freq[positive_freqs][fft_peaks] * 86400  # Convert to cycles per day
        fft_peak_magnitudes = fft_magnitude[fft_peaks]
         
        # Sort peaks by magnitude and select the top 5
        top_fft_peaks_indices = np.argsort(fft_magnitude[fft_peaks])[-5:]
        top_fft_peak_freqs = fft_freq[positive_freqs][fft_peaks][top_fft_peaks_indices] * 86400  # Convert to cycles per day
        top_fft_peak_magnitudes = fft_magnitude[fft_peaks][top_fft_peaks_indices]

        # Find the top five peaks in the power spectrum
        top_indices = np.argsort(power)[-5:]  # Get indices of the top 5 values
        top_frequencies = frequency_cpd[top_indices]
        top_powers = power[top_indices]

        # Highlight the fundamental frequencies from fft
        positive_freqs = fft_freq > 0
        max_amplitude_index = np.argmax(np.abs(fft_result)[positive_freqs])
        fundamental_frequency = fft_freq[positive_freqs][max_amplitude_index] * 86400  # Convert to cycles per day

        #Compute fundamental frequencies from the Lomb-Scargle method
        fund_frequency = frequency_cpd[np.argmax(power)]

        # Plot FFT results (optional)
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        plt.plot(fft_freq[fft_freq > 0] * 86400, np.abs(fft_result[fft_freq > 0]), label='FFT')  # FFT plot
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Frequency (cycles/day)')
        plt.ylabel('Amplitude')
        plt.title(f'Frequency Spectrum for {product_flag}')
        plt.grid(True)
        plt.legend()
        
        # Plot Lomb-Scargle periodogram
        plt.subplot(2, 1, 2)
        plt.plot(frequency_cpd, power, color='blue', label='Lomb-Scargle Periodogram')
        plt.xlabel('Frequency (cycles/day)')
        plt.ylabel('Power')
        plt.title(f'Lomb-Scargle Periodogram for {product_flag}')
        plt.grid(True)
        plt.legend()

        plt.show()
        
    return fft_results

#This function takes the values given from the fft_analysis and constructs a sinusoidal model
def construct_sinusoidal_model(fft_results, time_vectors):
    sinusoidal_models={}

    for product_flag, (fft_freq, fft_result) in fft_results.items():

        if not fft_result.any():
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
    return sinusoidal_models

def plot_comparison(data_vectors, sinusoidal_models, time_vectors):
    for product_flag in data_vectors.keys():
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
sorted_time_vectors, sorted_data_vectors=process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors)

# Plot the data
plot_data_vectors(sorted_data_vectors,sorted_time_vectors)

#perform fft on data
fft_results = perform_fft(sorted_data_vectors, sorted_time_vectors)

#perform lomb-scargle analysis
perform_lomb_scargle(data_vectors, time_vectors)
#construct sinusoidal model
sinusoidal_models=construct_sinusoidal_model(fft_results, sorted_time_vectors)

#plot sinusoidal model and data vectors form comparison
plot_comparison(sorted_data_vectors, sinusoidal_models, sorted_time_vectors)


