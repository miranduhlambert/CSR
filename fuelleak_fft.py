#Imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import glob
from datetime import datetime
import re


# Function that checks the Product Flag Sequences for Vectorizing data
def check_product_flag(product_flag):
    # Define the sequences you are looking for
    sequences = {
        'Temperatures': '01110100', #This and the following sequence for Van-Der-Waal data extractions
        'Pressures': '00000011'
        }
    
    #Using the key in the defined sequences, if the porduct flag is in the key return key 
    for key in sequences:
        if product_flag in sequences[key]:
            # If you find a match, exit the loop and return True
            return key
    #if no match is found, return None
    return None


#Function for filtering data to Bypass Yaml Header marker
def process_file_past_header(filename, marker, product_flag_index, data_indices, tank_num_index, time_s_index, time_ms_index, data_vectors, time_vector):
    #Define Variables for Lop Iteration
    marker_found= False
    reference_time_seconds=None

    # Initialize pressure storage
    last_pressure_values = {'1': None, '2': None}  # Assuming tanks are numbered 1 and 2
    last_pressure_times = {'1': None, '2': None}  # Store the corresponding times for pressure values
    
    with open(filename,'r') as file:
        for line in file:
            line=line.strip()
            if marker_found== True: # snippet to get all the data part the marker, do not have to set marker_found == True because it is defined with  
                columns = line.split() #Ensure the line is being treated as a string
                product_flag = columns[product_flag_index]
                product_flag_key = check_product_flag(product_flag)

                if product_flag_key:
                    tank_num = str(int(columns[tank_num_index]))
                        
                    if product_flag_key == 'Pressures':  # Check if this is a pressure line
                        # Update last seen pressure values
                        last_pressure_values[tank_num] = {
                            'tank_press': float(columns[data_indices[0]]),
                            'reg_press': float(columns[data_indices[1]])
                        }
                        if reference_time_seconds is None:
                                reference_time_seconds = float(columns[time_s_index])
                        time_ms = float(columns[time_ms_index]) / 1000000
                        time_seconds = float(columns[time_s_index]) - reference_time_seconds
                        last_pressure_times[tank_num] = time_seconds + time_ms

                        # print(f"Updated pressure values for tank {tank_num}: {last_pressure_values[tank_num]}")
                    elif product_flag_key == 'Temperatures':  # Check if this is a temperature line
                        if last_pressure_values[tank_num] is not None:
                            skin_temp = float(columns[data_indices[0]])
                            adap_temp = float(columns[data_indices[1]])
                                
                            # Use the last seen pressure values
                            tank_press = last_pressure_values[tank_num]['tank_press']
                            reg_press = last_pressure_values[tank_num]['reg_press']
                            press_time = last_pressure_times[tank_num]

                            if reference_time_seconds is None:
                                reference_time_seconds = float(columns[time_s_index])
                                
                            time_ms = float(columns[time_ms_index]) / 1000000  # Convert microseconds to seconds
                            time_seconds = float(columns[time_s_index]) - reference_time_seconds
                            time_combined = time_seconds + time_ms  # Combine seconds and milliseconds

                             # Append temperature data
                            data_vectors[f'tank{tank_num}']['temp_time'].append(time_combined)
                            data_vectors[f'tank{tank_num}']['skin_temp'].append(skin_temp)
                            data_vectors[f'tank{tank_num}']['adap_temp'].append(adap_temp)

                            # Append the pressure data corresponding to the last pressure before this temperature
                            data_vectors[f'tank{tank_num}']['press_time'].append(press_time)
                            data_vectors[f'tank{tank_num}']['tank_press'].append(tank_press)
                            data_vectors[f'tank{tank_num}']['reg_press'].append(reg_press)


                            # Debugging prints
                            # print(f"Appended time_combined: {time_combined}")
                            # print(f"Appended data for tank {tank_num}: Temp={skin_temp}, Adap Temp={adap_temp}, Tank Press={tank_press}, Reg Press={reg_press}, Time={time_combined}")

            elif marker in line:
                marker_found=True
                print(f"Marker found: {line}")

# # Defining Van der Waals modified equation for tanks
# def fTNK(n, R, P, T,a, b, v):
#     return (-a*b/(v**2))*n**3 + (a/v)*n**2 - (P*b + R*T)*n + P*v

# # Defining the derivative of Van der Waals modified equation for tanks
# def f_primeTNK(n, R, P, T,a, b, v):
#     return (-3*a*b/(v**2))*n**2 + (2*a/v)*n - (P*b) - (R*T)

# # Newton-Raphson method implementation
# def newton_raphson(f, f_prime, x0, P, T, tolerance=1e-8, max_iterations=100):
#     x = x0
#     for i in range(max_iterations):
#         fx = f(x, P, T)
#         fpx = f_prime(x, P, T)
#         x_new = x - fx / fpx
#         if abs(x_new - x) < tolerance:
#             return x_new
#         x = x_new
    

def extract_date_from_filename(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return match.group(0) if match else None

def extract_satellite_type_from_filename(filename):
    match = re.search(r'_(C|D)_', filename)
    return match.group(1) if match else None


def perform_fft(nested_data):
    fft_results = {}
   
    for satellite_type, tanks in nested_data.items():
        for tank,data in tanks.items():
            # Process temperature data
            temp_time = np.array(data['temp_time'])
            skin_temp = np.array(data['skin_temp'])
            adap_temp = np.array(data['adap_temp'])

            if len(temp_time)==0:
                print(f"No Temperature Data for {satellite_type} {tank}")
                continue

            # Calculate sampling frequency using time vectors
            sampling_intervals = np.diff(temp_time)
            average_sampling_interval = np.mean(sampling_intervals)
            sampling_frequency= 1/average_sampling_interval
            print(f'Sampling Frequency in Cycles/Day for {satellite_type} {tank} is : {sampling_frequency*86400}')
    
            #Perform FFT on Skin Temperature Data
            fft_result_skin= np.fft.fft(skin_temp)
            n=len(skin_temp)
            fft_freq=np.fft.fftfreq(n, d=average_sampling_interval)

            amplitude_skin=np.abs(fft_result_skin)*2/n
            phase_skin=np.angle(fft_result_skin)

            # Save FFT results for skin temperature
            fft_results[f'{satellite_type}_{tank}_skin_temp'] = (fft_freq, fft_result_skin, amplitude_skin, phase_skin)

            # Save FFT results in DataFrame
            fft_df_skin = pd.DataFrame({
                'Frequency (Cycles/Day)': fft_results[f'{satellite_type}_{tank}_skin_temp'][0] * 86400,
                'Amplitude': np.abs(fft_results[f'{satellite_type}_{tank}_skin_temp'][1]) * 2 / n,
                'Phase (Radians)': np.angle(fft_results[f'{satellite_type}_{tank}_skin_temp'][1])
            })
            
            fft_df_skin.to_csv(f'fft_{satellite_type}_{tank}_skin_temp_results.csv', index=False)
            print(f"FFT results saved for skin temperature of {satellite_type} {tank}")

            # Plot FFT results for skin temperature
            plt.figure(figsize=(14, 10))
            plt.plot(fft_freq * 86400, np.abs(fft_result_skin) * 2 / n, 'or', label='FFT (Skin Temp)')
            plt.axvline(x=sampling_frequency * 86400, color='r', linestyle='--', label=f'Sampling Frequency: {sampling_frequency * 86400} cycles/day')
            plt.axvline(x=sampling_frequency * 86400 / 2, color='g', linestyle='--', label=f'Nyquist Frequency: {sampling_frequency * 86400 / 2} cycles/day')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency (cycles/day)')
            plt.ylabel('Amplitude')
            plt.title(f'Frequency Spectrum for Skin Temperature of {satellite_type} {tank}')
            plt.grid(True)
            plt.legend()

            # Plot Frequency Versus Phase for skin temperature
            plt.figure()
            plt.plot(fft_freq * 86400, phase_skin, 'or')
            plt.xscale('log')
            plt.xlabel('Frequency (cycles/day)')
            plt.ylabel('Phase Angle')
            plt.title(f'Frequency Versus Phase for Skin Temperature of {satellite_type} {tank}')
            plt.grid(True)
            plt.legend()

            # Perform FFT on adap temperature data
            fft_result_adap = np.fft.fft(adap_temp)
            amplitude_adap = np.abs(fft_result_adap) * 2 / n
            phase_adap = np.angle(fft_result_adap)

            # Save FFT results for adap temperature
            fft_results[f'{satellite_type}_{tank}_adap_temp'] = (fft_freq, fft_result_adap, amplitude_adap, phase_adap)

            # Save FFT results in DataFrame
            fft_df_adap = pd.DataFrame({
                'Frequency (Cycles/Day)': fft_results[f'{satellite_type}_{tank}_adap_temp'][0] * 86400,
                'Amplitude': np.abs(fft_results[f'{satellite_type}_{tank}_adap_temp'][1]) * 2 / n,
                'Phase (Radians)': np.angle(fft_results[f'{satellite_type}_{tank}_adap_temp'][1])
            })

            fft_df_adap.to_csv(f'fft_{satellite_type}_{tank}_adap_temp_results.csv', index=False)
            print(f"FFT results saved for adap temperature of {satellite_type} {tank}")

            # Plot FFT results for adap temperature
            plt.figure(figsize=(14, 10))
            plt.plot(fft_freq * 86400, np.abs(fft_result_adap) * 2 / n, 'or', label='FFT (Adap Temp)')
            plt.axvline(x=sampling_frequency * 86400, color='r', linestyle='--', label=f'Sampling Frequency: {sampling_frequency * 86400} cycles/day')
            plt.axvline(x=sampling_frequency * 86400 / 2, color='g', linestyle='--', label=f'Nyquist Frequency: {sampling_frequency * 86400 / 2} cycles/day')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency (cycles/day)')
            plt.ylabel('Amplitude')
            plt.title(f'Frequency Spectrum for Adap Temperature of {satellite_type} {tank}')
            plt.grid(True)
            plt.legend()

            # Plot Frequency Versus Phase for adap temperature
            plt.figure()
            plt.plot(fft_freq * 86400, phase_adap, 'or')
            plt.xscale('log')
            plt.xlabel('Frequency (cycles/day)')
            plt.ylabel('Phase Angle')
            plt.title(f'Frequency Versus Phase for Adap Temperature of {satellite_type} {tank}')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()
            print(f"Plots generated for {satellite_type} {tank}")
        
    return fft_results



def analyze_files(vndrwl_files, marker, product_flag_index, data_indices, tank_num_index, time_s_index, time_ms_index):
    
    #initialize nested data structure and this function defininition will populate the data structure as the files are read
    nested_data={}
    time_vector=[]
    for filename in vndrwl_files:
        satellite_type = extract_satellite_type_from_filename(filename)
        if satellite_type not in nested_data:
            nested_data[satellite_type] = {'tank1': {}, 'tank2': {}}

        data_vectors = {
            'tank1': {
                'skin_temp': [],
                'adap_temp': [],
                'tank_press': [],
                'reg_press': [],
                'temp_time':[],
                'press_time':[]
            },
            'tank2': {
                'skin_temp': [],
                'adap_temp': [],
                'tank_press': [],
                'reg_press': [],
                'temp_time':[],
                'press_time':[]
            }
        }

    #Call the Function to Process the Data
        process_file_past_header(filename, marker, product_flag_index, data_indices, tank_num_index, time_s_index, time_ms_index, data_vectors, time_vector)
        nested_data[satellite_type] = data_vectors
    return nested_data


def plot_data(data_vectors):
    for satellite_type, tanks in data_vectors.items():
        for tank, data in tanks.items():
            # Plot temperature data
            if len(data['temp_time']) > 0:
                plt.figure(figsize=(12, 8))

                # Plot skin and adap temperatures
                plt.subplot(4, 1, 1)
                plt.plot(data['temp_time'], data['skin_temp'], label=f'Skin Temperature for {tank}')
                plt.plot(data['temp_time'], data['adap_temp'], label=f'Adap Temperature for {tank}')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Temperature (Celsius)')
                plt.title(f'Temperature Data for {tank} on GRACE-FO {satellite_type}')
                plt.legend()

                # Plot average temperature
                avg_temp = [(skin + adap) / 2 for skin, adap in zip(data['skin_temp'], data['adap_temp'])]
                plt.subplot(4, 1, 2)
                plt.plot(data['temp_time'], avg_temp, label=f'Average Temperature for {tank}', color='purple')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Average Temperature (Celsius)')
                plt.title(f'Average Temperature for {tank} on GRACE-FO {satellite_type}')
                plt.legend()

            # Plot pressure data
            if len(data['press_time']) > 0:
                plt.subplot(4, 1, 3)
                plt.plot(data['press_time'], data['tank_press'], label=f'Tank Pressure for {tank}')
                plt.plot(data['press_time'], data['reg_press'], label=f'Reg Pressure for {tank}')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Pressure (Bar)')
                plt.title(f'Pressure Data for {tank} on GRACE-FO {satellite_type}')
                plt.legend()

                # Plot average pressure
                avg_press = [(tank_p + reg_p) / 2 for tank_p, reg_p in zip(data['tank_press'], data['reg_press'])]
                plt.subplot(4, 1, 4)
                plt.plot(data['press_time'], avg_press, label=f'Average Pressure for {tank}', color='orange')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Average Pressure (Bar)')
                plt.title(f'Average Pressure for {tank} on GRACE-FO {satellite_type}')
                plt.legend()

            plt.tight_layout()
            plt.show()


              


#Define Variables for the Data Bypass
vndrwl_files=glob.glob(r'C:\data\Fuel Leak Data\TNK1A_*-*-*_*_04.txt') #Use glob to get all files that contain the data for Van-Der-Waals Calculations
# fuel_est_files=glob.glob(r'C:\data\Fuel Leak Data\MAS1A_*-*-*_*_04.txt') #use glob to get all files that contain the fuel mass estimates the tanks of both twin stellitss per hour
marker='# End of YAML header'
product_flag_index= 6 
data_indices_vndrwl= [7,8,9,10]                       # Adjust this index if the product flag is in a different column
tank_num_index=4
time_s_index= 0                         
time_ms_index= 1

# Initialize empty lists for storing data
time_vector = []        # Initialize empty time vector, should be the same for every product

# Analyze the files
nested_data = analyze_files(vndrwl_files, marker, product_flag_index, data_indices_vndrwl, tank_num_index, time_s_index, time_ms_index)

plot_data(nested_data)

fft_results = perform_fft(nested_data)

