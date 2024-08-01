#Imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob 
from datetime import datetime
import pandas as pd
import re

#Function that checks the Product Flag Sequences for Vectorizing data
def check_product_flag(product_flag):
    # Define the sequences you are looking for
    sequences = {
        'taicu':'00111100000000001000000001000000',  #bit 15 product flag; GF ACC_ICU temps; TICUN
        }
    
    #Using the key in the defined sequences, if the porduct flag is in the key return key 
    return next((key for key, value in sequences.items() if product_flag in value), None)
   
    #if no match is found, return None
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
def process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors):

    #Define Variables for Loop Iteration
    marker_found= False
    reference_time_seconds= None
    min_interval=0.1

    # Temporary storage for averaging
    temp_data_store = {key: [] for key in data_vectors.keys()}
    temp_time_store = {key: [] for key in time_vectors.keys()}

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
                        if reference_time_seconds is None:
                            reference_time_seconds = float(columns[0])

                        time_ms = float(columns[1]) / 1000000  # Convert microseconds to seconds
                        time_seconds = float(columns[0]) - reference_time_seconds
                        time_total_seconds = time_seconds + time_ms

                        if len(temp_time_store[product_flag_key]) == 0 or (time_total_seconds - temp_time_store[product_flag_key][-1]) <= min_interval:
                            # Store data temporarily
                            temp_data_store[product_flag_key].append(product_data)
                            temp_time_store[product_flag_key].append(time_total_seconds)
                        else:
                            # Average the stored data and reset temporary storage
                            avg_data = np.mean(temp_data_store[product_flag_key])
                            avg_time = np.mean(temp_time_store[product_flag_key])

                            data_vectors[product_flag_key].append(avg_data)
                            time_vectors[product_flag_key].append(avg_time)

                            # Clear temporary storage and add current data point
                            temp_data_store[product_flag_key] = [product_data]
                            temp_time_store[product_flag_key] = [time_total_seconds]

            elif marker in line:
                marker_found=True

        # Handle remaining data in temporary storage
        for key in temp_data_store:
            if len(temp_data_store[key]) > 0:
                avg_data = np.mean(temp_data_store[key])
                avg_time = np.mean(temp_time_store[key])
                data_vectors[key].append(avg_data)
                time_vectors[key].append(avg_time)


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

    nested_data = {
        'dates': [],
        'data_vectors': {},
        'time_vectors': {},
        'max_temps_per_day': {},
        'min_temps_per_day': {},
        'mean_temps_per_day': {},
        'frequencies_per_day': {},
        'amplitudes_per_day':{},
        'phases_per_day': {}
    }

    #Initializing Aggregation of all data 
    all_data_vectors = {}
    all_time_vectors = {}

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

        #Inialize empty dictionaries for each date if they dont exist
        if date not in nested_data['data_vectors']:
            nested_data['dates'].append(date)
            nested_data['data_vectors'][date] = {'taicu': []}
            nested_data['time_vectors'][date] = {'taicu': []}
            nested_data['max_temps_per_day'][date] = {'taicu': []}
            nested_data['min_temps_per_day'][date] = {'taicu': []}
            nested_data['mean_temps_per_day'][date] = { 'taicu': []}
            nested_data['frequencies_per_day'][date] = {'taicu': []}
            nested_data['amplitudes_per_day'][date]= {'taicu': []}
            nested_data['phases_per_day'][date]= {'taicu': []}
           
        #Call the file processing function to organize the data
        process_file_past_header(file_name, marker, product_flag_index, product_column_index, nested_data['data_vectors'][date], nested_data['time_vectors'][date])

        #Check the Sampling Intervals
        check_intervals(nested_data['time_vectors'][date])

        # Perform FFT on Data per day
        perform_fft(date, nested_data['data_vectors'][date], nested_data['time_vectors'][date], nested_data['frequencies_per_day'][date], nested_data['amplitudes_per_day'], nested_data['phases_per_day'])

        # Aggregate all data
        for date in nested_data['dates']:
            for key in nested_data['data_vectors'][date]:
                if key not in all_data_vectors:
                    all_data_vectors[key] = []
                    all_time_vectors[key] = []
                
                all_data_vectors[key].extend(nested_data['data_vectors'][date][key])
                all_time_vectors[key].extend(nested_data['time_vectors'][date][key])
    # Perform FFT on all aggregated data
    combined_date = 'All Data Combined'
    frequencies_per_date, amplitudes_per_date, phases_per_date = perform_fft(
        combined_date, all_data_vectors, all_time_vectors, 
        nested_data['frequencies_per_day'], nested_data['amplitudes_per_day'],
        nested_data['phases_per_day']
    )

    return nested_data

#Define Variables for the Data Analysis
file_list = glob.glob(r'C:\data\TAICU AHK1A 2023-08-01-31 C\AHK1A_2023-*-*_C_04.txt') #Adjust file pattern as needed C:\data\AHK1A_*
marker='# End of YAML header'
product_flag_index= 5                        # Adjust this index if the product flag is in a different column
product_column_index= 7                      # Adjust this index to the column where product data is located

#Call the function to analyze the files
nested_data = analyze_files(file_list, marker, product_flag_index, product_column_index)