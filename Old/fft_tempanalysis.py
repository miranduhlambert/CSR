#Imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob 
from datetime import datetime
import pandas as pd #used to put data into
import re

#Function that checks the Product Flag Sequences for Vectorizing data
def check_product_flag(product_flag):
    #Define the sequences you are looking for
    sequences = {
        'tesu': '00111100000000000100000001000000', #bit 14 product flag; Unknown on Offred; TSU_Y+
        'taicu':'00111100000000001000000001000000',  #bit 15 product flag; GF ACC_ICU temps; TICUN
        'tisu': '00111100000000010000000001000000',  #bit 16 product flag; Most likely GF1 ACC_Feeu; TSU_Y-
    }

    for key in sequences:
        if product_flag in sequences[key]:
            return key
    # Handle other flags or return None
    return None

def check_intervals(time_vectors):
    for product_flag in time_vectors:
        time_vector=np.array(time_vectors[product_flag])

        # Check if time vector is evenly spaced
        sampling_intervals = np.diff(time_vector)
        average_interval = np.mean(sampling_intervals)
        is_evenly_spaced = np.allclose(sampling_intervals, average_interval, rtol=1e-5, atol=1e-8)

        print(f"Product Flag: {product_flag}")
        print(f"Is evenly spaced: {is_evenly_spaced}")
        if not is_evenly_spaced:
            print(f"First 10 sampling intervals: {sampling_intervals[:10]}")
            print(f"Last 10 sampling intervals: {sampling_intervals[-10:]}")
            print(f"Total number of intervals: {len(sampling_intervals)}")


#Function for filtering data Bypass Yaml Header marker
def process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors, last_end_time):

    #Define Variables for Loop Iteration
    marker_found= False
    min_interval=0.1
    first_time_in_file= None
    current_end_time=None
   

    # Temporary storage for averaging redundant recordings
    temp_data_store = {key: [] for key in data_vectors.keys()}
    temp_time_store = {key: [] for key in time_vectors.keys()}

    with open(filename,'r') as file:
        for line in file:
            # snippet to get all the data part the marker, do not have to set marker_found == True because it is defined with
            if marker_found:   
                columns = line.split()
                
                if len(columns) > max(product_flag_index, product_column_index):
                    product_flag = columns[product_flag_index]
                    product_data = float(columns[product_column_index])
                    product_flag_key = check_product_flag(product_flag)

                    # print(f"Product Flag is {product_flag_key}")
                    if product_flag_key is None:
                        # print(f"Product flag key is None for flag: {product_flag}")  # Debugging
                        continue  # Skip this line if product_flag_key is None

                    if product_flag_key in data_vectors:
                        #Debugging Print
                        # print(f"Product Flag: {product_flag_key}, Product Data: {product_data}")  # Debugging

                        if first_time_in_file is None:
                            first_time_in_file= float(columns[0])
                            time_offset= last_end_time - first_time_in_file

                        #First time in file as: 
                            # rcvtime_intg:
                            # comment: 1st column
                            # coverage_content_type: referenceInformation
                            # long_name: Integer portion of time, in seconds past 12:00:00 noon of January 1, 2000 in OBC Time
                            # units: seconds

                        time_ms = float(columns[1]) / 1000000  # Convert microseconds to seconds
                        time_seconds = float(columns[0]) - first_time_in_file
                        time_total_seconds = time_seconds + time_ms 

                        if len(temp_time_store[product_flag_key]) == 0 or (time_total_seconds - temp_time_store[product_flag_key][-1]) <= min_interval:
                            temp_data_store[product_flag_key].append(product_data)
                            temp_time_store[product_flag_key].append(time_total_seconds)
                            # print(f"temp time store before averaging {temp_time_store}")
                        else:
                            if len(temp_data_store[product_flag_key]) > 0:
                                avg_data = np.mean(temp_data_store[product_flag_key])
                                avg_time=np.mean(temp_time_store[product_flag_key])
                                # print(f"Time Value after Averaging {avg_time}")
                                data_vectors[product_flag_key].append(avg_data)
                                time_vectors[product_flag_key].append(avg_time)

                            temp_data_store[product_flag_key] = [product_data]
                            temp_time_store[product_flag_key] = [time_total_seconds]

            elif marker in line:
                marker_found = True
                print(f"Marker found in line: {line.strip()}")  # Debugging


        for key in temp_data_store:
            if len(temp_data_store[key]) > 0:
                avg_data = np.mean(temp_data_store[key])
                avg_time = np.mean(temp_time_store[key])
                data_vectors[key].append(avg_data)
                time_vectors[key].append(avg_time)
    
        # Update the end time of the current file
        current_end_time = max([max(times) for times in time_vectors.values()]) if time_vectors else last_end_time    
    # check_intervals(time_vectors)
    return current_end_time
    

def perform_fft(date, data_vectors, time_vectors, frequencies_per_date, amplitudes_per_date, phases_per_date):
    for product_flag, data in data_vectors.items():
        
        if not date:
            continue
        
        time_vector = np.array(time_vectors[product_flag])

        if len(time_vector) < 2:
            print(f"Not enough data points for {product_flag} on {date} to perform FFT")
            continue

        sampling_intervals = np.diff(time_vector)
        if len(sampling_intervals) == 0:
            print(f"No sampling intervals found for {product_flag} on {date}")
            continue

        average_sampling_interval = np.mean(sampling_intervals)
        sampling_frequency = 1 / average_sampling_interval
        print(f'Sampling Frequency in Cycles/Day for {product_flag} is : {sampling_frequency*86400}')
    
        #Perform FFT on Data
        fft_result = np.fft.fft(data)
        n=len(data)
        fft_freq = np.fft.fftfreq(n, d=average_sampling_interval)
        frequencies_per_date[product_flag] = (fft_freq)

        # Calculate amplitudes and phases
        amplitude = np.abs(fft_result)
        phase = np.angle(fft_result)

        amplitudes_per_date[product_flag]= amplitude
        phases_per_date[product_flag]= phase

    return frequencies_per_date, amplitudes_per_date, phases_per_date



def find_temp_characteristics_per_day(data_vectors, time_vectors, max_temps_per_day, min_temps_per_day, mean_temps_per_day):
    
    for product_flag, time_vector in time_vectors.items():
        if not time_vector:
            continue
        data = data_vectors[product_flag]
        daily_mean_temp=np.mean(data)
        daily_max_temp = max(data) 
        daily_min_temp= min(data)

        #Update max_temps_per_day for the corresponding product flag
        max_temps_per_day[product_flag]= daily_max_temp
        min_temps_per_day[product_flag]= daily_min_temp
        mean_temps_per_day[product_flag]= daily_mean_temp

#Function to handle date extraction properly
def extract_date_from_filename(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return match.group(0) if match else None

def analyze_files(file_list, marker, product_flag_index, product_column_index):

    df = pd.DataFrame{
        'date': [],
        'data_vectors': {},
        'time_vectors': {},
        'max_temps_per_day': {},
        'min_temps_per_day': {},
        'mean_temps_per_day': {},
        'frequencies_per_day': {},
        'amplitudes_per_day':{},
        'phases_per_day': {}
    }

   # Initializing Aggregation of all data
    all_data_vectors = []
    all_time_vectors = []
    last_end_time = 0  # Initialize to track the end time of the last dataset
    global_reference_time = None
    
    for file_name in file_list:
    # Initialize empty vectors for storing data with coSrresponding product flag keys
        print(f"Processing file: {file_name}")
        
        # Extract date from filename
        date_str= extract_date_from_filename(file_name)
        if date_str is None:
            print("Date not found in file name:", file_name)
            continue

        # Convert date string to datetime object
        date = datetime.strptime(date_str, '%Y-%m-%d')

        #Initialize empty dictionaries for each date if they dont exist
        if date not in nested_data['data_vectors']:
            nested_data['dates'].append(date)
            nested_data['data_vectors'][date] = {'taicu': []}
            nested_data['time_vectors'][date] = {'taicu': []}
            nested_data['max_temps_per_day'][date] = {'taicu': []}
            nested_data['min_temps_per_day'][date] = {'taicu': []}
            nested_data['mean_temps_per_day'][date] = { 'taicu': []}
            nested_data['frequencies_per_day'][date] = { 'taicu': []}
            nested_data['amplitudes_per_day'][date] = { 'taicu': []}
            nested_data['phases_per_day'][date] = { 'taicu': []}

           
        #Call the file processing function to organize the data
        global_reference_time=process_file_past_header(file_name, marker, product_flag_index, product_column_index, nested_data['data_vectors'][date], nested_data['time_vectors'][date], last_end_time)
        
        # Compute temperature characteristics for each day
        find_temp_characteristics_per_day(
            nested_data['data_vectors'][date],
            nested_data['time_vectors'][date],
            nested_data['max_temps_per_day'][date],
            nested_data['min_temps_per_day'][date],
            nested_data['mean_temps_per_day'][date]
        )
        
    # Aggregate data for all days
    for date in nested_data['data_vectors']:
        for key in nested_data['data_vectors'][date]:
            data = nested_data['data_vectors'][date][key]
            time = nested_data['time_vectors'][date][key]

            # Check if time and data lengths match
            if len(time) != len(data):
                ValueError(f"Time and data length mismatch for {date}, key {key}.")

            # Adjust time vectors to continue from last_end_time
            if len(time) > 0:
                adjusted_time = [t + last_end_time for t in time]

                #Extend the final vectors
                all_data_vectors.extend(data)
                all_time_vectors.extend(adjusted_time)
                
                # Update the last_end_time
                last_end_time = adjusted_time[-1] if adjusted_time else last_end_time
        
     # Debugging: Print all time vectors before duplicate check
    print(f"All time vectors before duplicate check: {all_time_vectors}")

    # Final check to ensure no duplicate times
    time_set = set()
    duplicates = []
    for time in all_time_vectors:
        if time in time_set:
            duplicates.append(time)
        time_set.add(time)

    if duplicates:
        print(f"Duplicate times found: {duplicates}")
        raise ValueError("Duplicate times detected in the final aggregated time vector.")
        
    # Convert aggregated lists to numpy arrays for easier manipulation
    print(len(all_data_vectors))
    print(len(all_time_vectors))
    all_data_vectors = np.array(all_data_vectors)
    all_time_vectors = np.array(all_time_vectors)
    print(all_time_vectors)

    # Perform FFT on all aggregated data
    sampling_intervals=np.diff(all_time_vectors)
    average_sampling_interval= np.mean(sampling_intervals)
    print(sampling_intervals)
    sampling_frequency=1/4.799
    n=len(all_data_vectors)
    fft_result_all_data=np.fft.fft(all_data_vectors)
    all_frequencies_vectors=np.fft.fftfreq(n,d=average_sampling_interval)
    all_amplitudes_vectors=np.abs(fft_result_all_data)
    all_phase_vectors=np.angle(fft_result_all_data)


    # Plot FFT results for all aggregated data
    plt.figure(figsize=(14, 10))
    plt.plot(all_frequencies_vectors*86400, np.abs(all_amplitudes_vectors)*2/n, 'or', label='FFT')  # FFT plot; 5400 seconds is about the time of one revolution
    plt.axvline(x=1/np.mean(np.diff(all_time_vectors))*86400, color='r', linestyle='--', label=f'Sampling Frequency: {1 / np.mean(np.diff(all_time_vectors)) * 86400} Hz')
    plt.axvline(x=1 / (2 * np.mean(np.diff(all_time_vectors))) * 86400, color='g', linestyle='--', label=f'Nyquist Frequency: {1 / (2 * np.mean(np.diff(all_time_vectors)))} Hz')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (cycles/day)')
    plt.ylabel('Amplitude')
    plt.title(f'Frequency Spectrum for ')
    plt.grid(True)
    plt.legend()

    #Plot Frequency Versus Phase for all aggregated data
    plt.figure()
    plt.plot(all_frequencies_vectors*86400, all_phase_vectors, 'or')
    plt.xscale('log')
    plt.xlabel('Frequency (cycles/day)')
    plt.ylabel('Phase Angle')
    plt.title(f'Frequency Versus Phase')
    plt.grid(True)
    plt.legend()

    #Plot Data
    plt.figure()
    plt.plot(all_time_vectors, all_data_vectors, 'or', label=f'Original Data for GRACE-FO C: 2024-05-01')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Temperature (Celsius)')
    plt.title(f'Original Data for')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return nested_data

#Define Variables for the Data Analysis
file_list = glob.glob(r'C:\data\TAICU AHK1A 2023-08-01-31 C\AHK1A_2023-*-*_C_04.txt') #Adjust file pattern as needed C:\data\AHK1A_*
marker='# End of YAML header'
product_flag_index= 5                        # Adjust this index if the product flag is in a different column
product_column_index= 7                      # Adjust this index to the column where product data is located

#Call the function to analyze the files
nested_data = analyze_files(file_list, marker, product_flag_index, product_column_index)