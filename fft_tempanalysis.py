#Imports
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use Tk'Agg backend
import matplotlib.dates as mdates
import numpy as np
import glob 
import pprint


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
def process_file_past_header(date, filename, marker, product_flag_index, product_column_index, data_vectors, time_vectors):

    #Define Variables for Loop Iteration
    marker_found= False
    reference_time_seconds= None

    with open(filename,'r') as file:

        for line in file:

            if marker_found: # snippet to get all the data part the marker, do not have to set marker_found == True because it is defined with  
        
                columns = line.split()

                # Ensure the line has at least 5 columns to avoid index error
                if len(columns) > max(product_flag_index, product_column_index):
                    
                    #obtain Product Flag and Product data
                    product_flag = columns[product_flag_index]
                    product_data= float(columns[product_column_index])

                    #Call check_product_flag to get the product flag key
                    product_flag_key=check_product_flag(product_flag) 

                    # if product flag key is found
                    if product_flag_key in data_vectors:   
                                           
                            #Append data to the corresponding key in data_vectors
                            data_vectors[product_flag_key].append(float(product_data))

                            #Extract the reference time from the first line of each file
                            if reference_time_seconds is None:
                                reference_time_seconds= float(columns[0])

                            time_ms = float(columns[1]) / 1000  # Convert milliseconds to seconds
                            time_seconds = float(columns[0]) - reference_time_seconds  # Subtract the reference time point
                            time_vectors[product_flag_key].append(time_seconds + time_ms)  # Combine seconds and milliseconds

            elif marker in line:
                marker_found=True


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
        fft_freq = np.fft.fftfreq(len(data_detrended), d=average_sampling_interval)
        frequencies_per_date[product_flag] = (fft_freq, fft_result)


def find_temp_characteristics_per_day(date,data_vectors, time_vectors, max_temps_per_day, min_temps_per_day, temp_variations_per_day):
    
    for product_flag, time_vector in time_vectors.items():
        if not time_vector:
            continue
        data = data_vectors[product_flag]
        daily_max_temp = max(data) 
        daily_min_temp= min(data)
        variation_temp= abs(daily_max_temp - daily_min_temp)

        #Update max_temps_per_day for the corresponding product flag
        max_temps_per_day[product_flag]= daily_max_temp
        min_temps_per_day[product_flag]= daily_min_temp
        temp_variations_per_day[product_flag]= variation_temp



def analyze_files(file_list, marker, product_flag_index, product_column_index):

    nested_data = {
        'dates': [],
        'data_vectors': {},
        'time_vectors': {},
        'max_temps_per_day': {},
        'min_temps_per_day': {},
        'temp_variations_per_day': {},
        'frequencies_per_date': {},
        'fund_freq_per_date': {}
    }

    for file_name in file_list:
    # Initialize empty vectors for storing data with coSrresponding product flag keys
        # Extract date from filename
        date = None
        parts = file_name.split('_')
        for part in parts:
            if part.startswith('2022'):
                date = part
                break
        
        if date is None:
            print("Date not found in file name:", file_name)
            continue

        #Inialize empty dictionaries for each date if they dont exist
        if date not in nested_data['data_vectors']:
            nested_data['dates'].append(date)
            nested_data['data_vectors'][date] = {'tesu': [], 'taicu': [], 'tisu': [], 'tcicu': []}
            nested_data['time_vectors'][date] = {'tesu': [], 'taicu': [], 'tisu': [], 'tcicu': []}
            nested_data['max_temps_per_day'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}, 'tcicu': {}}
            nested_data['min_temps_per_day'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}, 'tcicu': {}}
            nested_data['temp_variations_per_day'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}, 'tcicu': {}}
            nested_data['frequencies_per_date'][date] = {'tesu': {}, 'taicu': {}, 'tisu': {}, 'tcicu': {}}
            nested_data['fund_freq_per_date'][date]= {'tesu': {}, 'taicu': {}, 'tisu': {}, 'tcicu': {}}
           
        #Call the file processing function to organize the data
        process_file_past_header(date, file_name, marker, product_flag_index, product_column_index, nested_data['data_vectors'][date], nested_data['time_vectors'][date])

        # Call the perform_fft function
        perform_fft(date, nested_data['data_vectors'][date], nested_data['time_vectors'][date], nested_data['frequencies_per_date'][date])

        for product_flag, (fft_freq, fft_result) in nested_data['frequencies_per_date'][date].items():
            positive_freqs = fft_freq > 0
            max_amplitude_index = np.argmax(np.abs(fft_result)[positive_freqs])
            fundamental_frequency = fft_freq[positive_freqs][max_amplitude_index] * 86400
            nested_data['fund_freq_per_date'][date][product_flag] = fundamental_frequency
        
        find_temp_characteristics_per_day(date, nested_data['data_vectors'][date], nested_data['time_vectors'][date], nested_data['max_temps_per_day'][date], nested_data['min_temps_per_day'][date], nested_data['temp_variations_per_day'][date])

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
        temp_variations = []

        for date in dates:
            if product_flag in nested_data['min_temps_per_day'][date] and \
               product_flag in nested_data['max_temps_per_day'][date] and \
               product_flag in nested_data['temp_variations_per_day'][date]:
                min_temp = nested_data['min_temps_per_day'][date][product_flag]
                max_temp = nested_data['max_temps_per_day'][date][product_flag]
                temp_variation = nested_data['temp_variations_per_day'][date][product_flag]
                
                min_temps.append(min_temp)
                max_temps.append(max_temp)
                temp_variations.append(temp_variation)

        ax = axes[i] if num_products > 1 else axes
        ax.plot(dates[:len(min_temps)], min_temps, label='Min Temp')
        ax.plot(dates[:len(max_temps)], max_temps, label='Max Temp')
        ax.plot(dates[:len(temp_variations)], temp_variations, label='Temp Variation')
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




#Define Variables for the Data Analysis

file_list = glob.glob(r'C:\data\AHK1A 2022_07_01-08_31 D\AHK1A_*') #Adjust file pattern as needed C:\data\AHK1A_*

# print("Files found:")
# print(file_list)

marker='# End of YAML header'
product_flag_index= 5                        # Adjust this index if the product flag is in a different column
product_column_index= 7                      # Adjust this index to the column where product data is located

#Call the function to analyze the files
nested_data = analyze_files(file_list, marker, product_flag_index, product_column_index)


# Call the functions the plot the data
#plot_data_vectors(nested_data)
plot_frequency_over_time(nested_data)
plot_temperature_statistics(nested_data)