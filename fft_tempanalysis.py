#Imports
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use TkAgg backend
import numpy as np
import glob 


#Function that checks the Product Flag Sequences for Vectorizing data
def check_product_flag(product_flag):
    # Define the sequences you are looking for
    sequences = {
        'tesu': '00111100000000000100000001000000', #bit 14 product flag; Unknown on Offred; TSU_Y+
        'taicu':'00111100000000001000000001000000',  #bit 15 product flag; GF1 ACC Hk N ICU temps; TICUN
        'tisu': '00111100000000010000000001000000',  #bit 16 product flag; Unknown on Offred; TSU_Y-
        'tcicu':'00111100000100000000000001000000'   #bit 20 product flag; GF1 ACC Hk R ICU temps ; TICUR
        }
    #Using the key in the defined sequences, if the porduct flag is in the key return key 
    for key in sequences:
        if product_flag in sequences[key]:
            # If you find a match, exit the loop and return True
            return key
    #if no match is found, return None
    return None

#Function for filtering data Bypass Yaml Header marker
def process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors):

    marker_found= False
    reference_time_seconds= None

    with open(filename,'r') as file:

        for line in file:

            # Debugging output to trace the lines being read

            # print(f"Reading line: {line.strip()}")

            if marker_found: # snippet to get all the data part the marker, do not have to set marker_found == True because it is defined with  
        
                columns = line.split()

                # Ensure the line has at least 5 columns to avoid index errors
                
                if len(columns) > max(product_flag_index, product_column_index):
                    
                    product_flag = columns[product_flag_index]  # Extract the fifth column
                    product_data= columns[product_column_index]

                    #Call check_product_flag to get the product flag key
                    product_flag_key=check_product_flag(product_flag) 

                    if product_flag_key in data_vectors:                  # if product flag key is found
                            #Append data to the corresponding key in data_vectors
                            data_vectors[product_flag_key].append(float(product_data))

                            if reference_time_seconds is None:
                                #Extract the reference time from the first line of each file
                                reference_time_seconds= float(columns[0])

                            time_ms = float(columns[1]) / 1000  # Convert milliseconds to seconds
                            time_seconds = float(columns[0]) - reference_time_seconds  # Subtract the reference time point
                            time_vectors[product_flag_key].append(time_seconds + time_ms)  # Combine seconds and milliseconds

            elif marker in line:
                marker_found=True
                # print(f"Marker found: {marker}")


#Function that performs the Fast Fourier Transform Analysis and plots the results
def perform_fft(data_vectors,time_vectors):

    #Define the vector for results 
    fft_results = {}

    #For each product_flag key for columns (data) in data_vectors
    for product_flag, data in data_vectors.items():

        # Calculate sampling frequency using specified time vector
        time_vector = time_vectors[product_flag]

        if len(time_vector) < 2:
            print(f"Not enough data points for {product_flag} to perform FFT")
            continue

        sampling_intervals = np.diff(time_vector)
        if len(sampling_intervals) == 0:
            print(f"No sampling intervals found for {product_flag}")
            continue

        average_sampling_interval = np.mean(sampling_intervals)
        sampling_frequency = 1 / average_sampling_interval

        print(f"Processing {product_flag}:")
        print(f"  Number of data points: {len(data)}")
        print(f"  Number of time points: {len(time_vector)}")
        print(f"  Average sampling interval: {average_sampling_interval}")
        print(f"  Sampling frequency: {sampling_frequency}")

        # Detrend the data by subtracting the mean
        data_detrended = data - np.mean(data)

        # Perform FFT
        fft_result = np.fft.fft(data_detrended)
        fft_freq = np.fft.fftfreq(len(data_detrended), d=average_sampling_interval)

        # Save FFT results
        fft_results[product_flag] = (fft_freq, fft_result)

        positive_freqs = fft_freq > 0   
        # Plot FFT results (positive frequencies only)
        # plt.figure()
        # plt.plot(fft_freq[positive_freqs] * 86400, np.abs(fft_result)[positive_freqs])  # Convert to cycles per day
        # plt.xlabel('Frequency (cycles/day)')
        # plt.ylabel('Amplitude')
        # plt.title(f'Frequency Spectrum for {product_flag}')
        # plt.grid(True)
        # plt.show()

        # Highlight the fundamental frequencies
        max_amplitude_index = np.argmax(np.abs(fft_result)[positive_freqs])
        fundamental_frequency = fft_freq[positive_freqs][max_amplitude_index] * 86400  # Convert to cycles per day
        
        print(f"Fundamental frequency for {product_flag}: {fundamental_frequency:.2f} cycles/day")
        
    return fft_results

def find_max_temp_per_day(date, data_vectors, time_vectors, max_temps_per_day):
    
    for product_flag, time_vector in time_vectors.items():
        if not time_vector:
            continue
        data = data_vectors[product_flag]
    
        daily_max_temp = max(data) if data else None # Initialize daily_max_temps for each product_flag

      # Update max_temps_all_files for the corresponding product flag and date
        if date in max_temps_per_day:
            max_temps_per_day[date][product_flag] = daily_max_temp
        else:
            #initialize the date entry in max_temps_per_day with an empty dictionary
            max_temps_per_day[date] = {}
            #Update max_temps_per_day for the corresponding product flag
            max_temps_per_day[date][product_flag]= daily_max_temp
    return max_temps_per_day


def analyze_files(file_list, marker, product_flag_index, product_column_index):

    nested_data = {
        'dates': [],
        'data_vectors': {},
        'time_vectors': {},
        'max_temps_per_day': {},
        'frequencies_per_date': {},
        'max_temps_all_files': {}
    }

    for file_name in file_list:
    # Initialize empty vectors for storing data with corresponding product flag keys
        #Extract date from filename
        date=file_name.split('_')[1]

        #Inialize empty dictionaries for each date if they dont exist
        if date not in nested_data['data_vectors']:
            nested_data['dates'].append(date)
            nested_data['data_vectors'][date] = {'tesu': [], 'taicu': [], 'tisu': [], 'tcicu': []}
            nested_data['time_vectors'][date] = {'tesu': [], 'taicu': [], 'tisu': [], 'tcicu': []}
            nested_data['max_temps_per_day'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}, 'tcicu': {}}
            nested_data['frequencies_per_date'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}, 'tcicu': {}}

 
        #Call the file processing function to organize the data
        process_file_past_header(file_name, marker, product_flag_index, product_column_index, nested_data['data_vectors'][date], nested_data['time_vectors'][date])

        #Call the function to perform Fast Fourier Transform operations on data
        fft_results = perform_fft(nested_data['data_vectors'][date], nested_data['time_vectors'][date])

        for product_flag, (fft_freq, fft_result) in fft_results.items():
            positive_freqs = fft_freq > 0
            max_amplitude_index = np.argmax(np.abs(fft_result)[positive_freqs])
            fundamental_frequency = fft_freq[positive_freqs][max_amplitude_index] * 86400
            print(f"Fundamental frequency for {product_flag} on {date}: {fundamental_frequency:.2f} cycles/day")  # Debug print
            nested_data['frequencies_per_date'][date][product_flag]=fundamental_frequency

        nested_data['max_temps_per_day']=find_max_temp_per_day(date, nested_data['data_vectors'][date], nested_data['time_vectors'][date], nested_data['max_temps_per_day'])


    return nested_data

#Function that plots the data vectors
def plot_data_vectors(nested_data):

    #for each product flag and data vector in the data vectors
    for date, product_flags in nested_data['data_vectors'].items():

        for product_flag, data in product_flags.items():
            #If statement to say that there is no data to plot for a specific product flag
            if not data:
                print(f"No data to plot for {product_flag}")
                continue

            #Use the time vector for the corresponding product flag
            time_vector = nested_data['time_vectors'][date][product_flag]

            #Plot the data
            plt.figure()
            plt.plot(time_vector, data, label=product_flag)
            plt.xlabel('Time of Day (seconds)')
            plt.ylabel('Temperature (Celsius)')
            plt.title(f'{product_flag} data on {date}')
            plt.legend()
            plt.grid(True)
            plt.show()

def plot_frequency_over_time(nested_data):
    # Extract dates
    dates = list(nested_data['frequencies_per_date'].keys())

    # Iterate over product flags
    for product_flag in nested_data['frequencies_per_date'][dates[0]].keys():
        # Extract frequencies for the current product flag
        frequencies = [nested_data['frequencies_per_date'][date][product_flag] for date in dates]

        # Plot frequencies over time for the current product flag
        plt.figure()
        plt.plot(dates, frequencies, label=product_flag)
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.title(f'{product_flag} Frequency Over Time')
        plt.legend()
        plt.show()

def plot_max_temps(nested_data):
    dates = list(nested_data['max_temps_per_day'].keys())
    for product_flag in nested_data['max_temps_per_day'][dates[0]].keys():
        max_temps = [nested_data['max_temps_per_day'][date][product_flag] for date in dates]

        plt.figure()
        plt.plot(dates, max_temps, label=product_flag)
        plt.xlabel('Date')
        plt.ylabel('Max Temperature (Celsius)')
        plt.title(f'Max Temperature for {product_flag} Over Time')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

# def check_flags_in_files(file_list):
#     flag_sequences = {
#         'tesu': '00111100000000000100000001000000',
#         'taicu': '00111100000000001000000001000000',
#         'tisu': '00111100000000010000000001000000',
#         'tcicu': '00111100000100000000000001000000'
#     }
#     flag_counts = {flag: 0 for flag in flag_sequences}
#     for file_name in file_list:
#         print(f"Scanning file for flags: {file_name}")
#         with open(file_name, 'r') as file:
#             for line in file:
#                 for flag, sequence in flag_sequences.items():
#                     if sequence in line:
#                         flag_counts[flag] += 1
    
#     print("Flag presence in files:")
#     for flag, count in flag_counts.items():
#         print(f"  {flag}: {count} occurrences")




#Define Variables for the Data Bypass

file_list = glob.glob(r'C:\data\AHK1A_*.txt') #Adjust file pattern as needed
print("Files found:")
print(file_list)
marker='# End of YAML header'
product_flag_index= 5                        # Adjust this index if the product flag is in a different column
product_column_index= 7                      # Adjust this index to the column where product data is located



#Call the functions to find frequency over time
# check_flags_in_files(file_list)  # Add this line to check for flag presence in the files
nested_data = analyze_files(file_list, marker, product_flag_index, product_column_index)

# Call the functions the plot the data
#plot_data_vectors(nested_data)
plot_frequency_over_time(nested_data)
plot_max_temps(nested_data)
