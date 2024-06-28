#Imports
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use Tk'Agg backend
import matplotlib.dates as mdates
import numpy as np
import glob 
from datetime import datetime


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
def process_file_past_header(date, filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors):

    #Define Variables for Loop Iteration
    marker_found= False
    reference_time_seconds= None
    min_interval=0.1

    # Temporary storage for averaging
    temp_data_store = {key: [] for key in ['tesu', 'taicu', 'tisu']}
    temp_time_store = {key: [] for key in ['tesu', 'taicu', 'tisu']}

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


def perform_fft(date, data_vectors, time_vectors, frequencies_per_date):
        
    for product_flag, data in data_vectors.items():
        time_vector = time_vectors[product_flag]

        if len(time_vector) < 2:
            print(f"Not enough data points for {product_flag} on {date} to perform FFT")
            continue

        sampling_intervals = np.diff(time_vector)
        if len(sampling_intervals) == 0:
            print(f"No sampling intervals found for {product_flag} on {date}")
            continue

        average_sampling_interval = np.mean(sampling_intervals)
        sampling_frequency = 1 / average_sampling_interval
        data_detrended = data - np.mean(data)
        fft_result = np.fft.fft(data_detrended)
        # fft_freq = np.fft.fftfreq(len(data_detrended), d=average_sampling_interval)
        fft_freq = np.fft.fftfreq(len(data_detrended), d=average_sampling_interval)
        frequencies_per_date[product_flag] = (fft_freq, fft_result)

    return frequencies_per_date



def find_temp_characteristics_per_day(date,data_vectors, time_vectors, max_temps_per_day, min_temps_per_day, mean_temps_per_day):
    
    for product_flag, time_vector in time_vectors.items():
        if not time_vector:
            continue
        data = data_vectors[product_flag]
        daily_mean_temp=np. mean(data)
        daily_max_temp = max(data) 
        daily_min_temp= min(data)

        #Update max_temps_per_day for the corresponding product flag
        max_temps_per_day[product_flag]= daily_max_temp
        min_temps_per_day[product_flag]= daily_min_temp
        mean_temps_per_day[product_flag]= daily_mean_temp



def analyze_files(file_list, marker, product_flag_index, product_column_index):

    nested_data = {
        'dates': [],
        'data_vectors': {},
        'time_vectors': {},
        'max_temps_per_day': {},
        'min_temps_per_day': {},
        'mean_temps_per_day': {},
        'frequencies_per_date': {},
        'fund_freq_per_date': {}
    }

    for file_name in file_list:
    # Initialize empty vectors for storing data with coSrresponding product flag keys
        # Extract date from filename
        date_str = None
        parts = file_name.split('_')
        for part in parts:
            if part.startswith('2022'): #change this based on the data
                date_str = part
                break
        
        if date_str is None:
            print("Date not found in file name:", file_name)
            continue

        # Convert date string to datetime object
        date = datetime.strptime(date_str, '%Y-%m-%d')

        #Inialize empty dictionaries for each date if they dont exist
        if date not in nested_data['data_vectors']:
            nested_data['dates'].append(date)
            nested_data['data_vectors'][date] = {'tesu': [], 'taicu': [], 'tisu': []}
            nested_data['time_vectors'][date] = {'tesu': [], 'taicu': [], 'tisu': []}
            nested_data['max_temps_per_day'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}}
            nested_data['min_temps_per_day'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}}
            nested_data['mean_temps_per_day'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}}
            nested_data['frequencies_per_date'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}}
            nested_data['fund_freq_per_date'][date]= {'tesu': {}, 'taicu': {}, 'tisu': {}}
           
        #Call the file processing function to organize the data
        process_file_past_header(date, file_name, marker, product_flag_index, product_column_index, nested_data['data_vectors'][date], nested_data['time_vectors'][date])

        #Check the Sampling Intervals
        #check_intervals(nested_data['time_vectors'][date])

        # Call the perform_fft function
        perform_fft(date, nested_data['data_vectors'][date], nested_data['time_vectors'][date], nested_data['frequencies_per_date'][date])

        for product_flag, (fft_freq, fft_result) in nested_data['frequencies_per_date'][date].items():
            positive_freqs = fft_freq > 0
            max_amplitude_index = np.argmax(np.abs(fft_result)[positive_freqs])
            fundamental_frequency = fft_freq[positive_freqs][max_amplitude_index] * 86400
            nested_data['fund_freq_per_date'][date][product_flag] = fundamental_frequency
        
        find_temp_characteristics_per_day(date, nested_data['data_vectors'][date], nested_data['time_vectors'][date], nested_data['max_temps_per_day'][date], nested_data['min_temps_per_day'][date], nested_data['mean_temps_per_day'][date])

    return nested_data

def plot_frequency_over_time(nested_data):
    
    # Extract dates
    dates = nested_data['dates']
    
    #Getting number of products to ensure time and data 
    num_products = len(nested_data['fund_freq_per_date'][dates[0]])
    fig, axes = plt.subplots(nrows=1, ncols=num_products, figsize=(15, 5))  # Adjust figsize as needed

    # Iterate over product flags
    for i, product_flag in enumerate(nested_data['fund_freq_per_date'][dates[0]]):
        
        frequencies=[]

        # Extract frequencies for the current product flag
        for date in dates:
            if product_flag in nested_data['fund_freq_per_date'][date]:
                frequency = nested_data['fund_freq_per_date'][date][product_flag]
                frequencies.append(frequency)

        ax = axes[i] if num_products > 1 else axes
        ax.plot(dates[:len(frequencies)], frequencies, label='Fundamental Frequency')
        ax.set_xlabel('Date')
        ax.set_ylabel('Fundamental Frequency in /Day')
        ax.set_title(f'Fundamental Frequency for {product_flag} Over Time')
        ax.legend()
        ax.grid(True)

        # Formatting the x-axis for each subplot
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Change the interval as needed
        ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

# Plot Min, Max, and Temperature Variations per Day
def plot_temperature_statistics(nested_data):
    dates = nested_data['dates']
    num_products = len(nested_data['max_temps_per_day'][dates[0]])
    fig, axes = plt.subplots(nrows=1, ncols=num_products, figsize=(15, 5))  # Adjust figsize as needed

    for i, product_flag in enumerate(nested_data['max_temps_per_day'][dates[0]]):
        min_temps = []
        max_temps = []
        mean_temps = []

        for date in dates:
            if product_flag in nested_data['min_temps_per_day'][date] and \
               product_flag in nested_data['max_temps_per_day'][date] and \
               product_flag in nested_data['mean_temps_per_day'][date]:
                min_temp = nested_data['min_temps_per_day'][date][product_flag]
                max_temp = nested_data['max_temps_per_day'][date][product_flag]
                mean_temp = nested_data['mean_temps_per_day'][date][product_flag]
                
                min_temps.append(min_temp)
                max_temps.append(max_temp)
                mean_temps.append(mean_temp)

        ax = axes[i] if num_products > 1 else axes
        ax.plot(dates[:len(min_temps)], min_temps, label='Min Temp')
        ax.plot(dates[:len(max_temps)], max_temps, label='Max Temp')
        ax.plot(dates[:len(mean_temps)], mean_temps, label='Mean Temp')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (Celsius)')
        ax.set_title(f'Temperature Statistics for {product_flag} Over Time')
        ax.legend()
        ax.grid(True)
        
        # Formatting the x-axis for each subplot
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Change the interval as needed
        ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

#Define Variables for the Data Analysis
file_list = glob.glob(r'C:\data\AHK1A 2022_07_01-08_31 D\AHK1A_*') #Adjust file pattern as needed C:\data\AHK1A_*
marker='# End of YAML header'
product_flag_index= 5                        # Adjust this index if the product flag is in a different column
product_column_index= 7                      # Adjust this index to the column where product data is located

#Call the function to analyze the files
nested_data = analyze_files(file_list, marker, product_flag_index, product_column_index)

# Call the functions the plot the data
#plot_frequency_over_time(nested_data)
plot_temperature_statistics(nested_data)