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
        'Pressures': '00000011',
        'Fuel Mass Estimates': '11000000' #Fuel Estimate Files
        }
    
    #Using the key in the defined sequences, if the porduct flag is in the key return key 
    for key in sequences:
        if product_flag in sequences[key]:
            # If you find a match, exit the loop and return True
            return key
   
    #if no match is found, return None
    return None


#Function for filtering data to Bypass Yaml Header marker
def process_file_past_header(filename, marker, product_flag_index, data_indices, time_s_index, time_ms_index, data_vectors, time_vector):
    #Define Variables for Lop Iteration
    marker_found= False
    reference_time_seconds=None

    with open(filename,'r') as file:

        for line in file:

            if marker_found: # snippet to get all the data part the marker, do not have to set marker_found == True because it is defined with  
        
                columns = line.split()

              # Ensure the line has at least 5 columns to avoid index error
                if len(columns) > max(product_flag_index, *data_indices, time_s_index, time_ms_index):
                    
                    product_flag = columns[product_flag_index]

                    product_flag_key = check_product_flag(product_flag)
                    
                    if product_flag_key== 'Fuel Mass Estimates'
                        

                    if product_flag_key:
                        reg_press = float(columns[data_indices[0]])
                        skin_temp=float(columns[data_indices[1]])
                        skin_temp_r=float(columns[data_indices[2]])
                        boss_fixed=float(columns[data_indices[3]])

                        if reference_time_seconds is None:
                            reference_time_seconds = float(columns[time_s_index])
                    
                        time_ms = float(columns[time_ms_index]) / 1000000  # Convert microseconds to seconds
                        time_seconds = float(columns[time_s_index]) - reference_time_seconds
                        time_combined= time_seconds + time_ms  # Combine seconds and milliseconds

                        #Append data    
                        data_vectors['reg_press'].append(float(reg_press))
                        data_vectors['skin_temp'].append(float(skin_temp))
                        data_vectors['skin_temp_r'].append(float(skin_temp_r))
                        data_vectors['boss_fixed'].append(float(boss_fixed))
                        time_vector.append(time_combined)
                        
                        print(f"Appended data for product_flag {product_flag_key}")

            elif marker in line:
                marker_found=True
                print("Marker found in file")

# Defining Van der Waals modified equation for tanks
def fTNK(n, R, P, T,a, b, v, n):
    return (-a*b/(v**2))*n**3 + (a/v)*n**2 - (P*b + R*T)*n + P*v

# Defining the derivative of Van der Waals modified equation for tanks
def f_primeTNK(n, R, P, T,a, b, v, n):
    return (-3*a*b/(v**2))*n**2 + (2*a/v)*n - (P*b) - (R*T)

# Newton-Raphson method implementation
def newton_raphson(f, f_prime, x0, P, T, tolerance=1e-8, max_iterations=100):
    x = x0
    for i in range(max_iterations):
        fx = f(x, P, T)
        fpx = f_prime(x, P, T)
        x_new = x - fx / fpx
        if abs(x_new - x) < tolerance:
            return x_new
        x = x_new
    

def extract_date_from_filename(filename):
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    return match.group(0) if match else None

def extract_satellite_type_from_filename(filename):
    match = re.search(r'_(C|D)_', filename)
    return match.group(1) if match else None


def perform_fft(data_vectors, time_vector):
    fft_results = {}
   
    for product_flag, data in data_vectors.items():
        
        if not data:
            print(f"No data for product flag {product_flag}")
            continue
        
        # Calculate sampling frequency using time vectors
        time_vector = np.array(time_vector)
        data=np.array(data)
        sampling_intervals = np.diff(time_vector)
        average_sampling_interval = np.mean(sampling_intervals)
        sampling_frequency= 1/average_sampling_interval
        print(f'Sampling Frequency in Cycles/Day for {product_flag} is : {sampling_frequency*86400}')
    
       # Apply Bartlett window to the data
        # window = np.kaiser(len(data),0)
        # windowed_data = data * window
        
        # Perform FFT on windowed data
        fft_result = np.fft.fft(data)
        n = len(data)
        fft_freq = np.fft.fftfreq(n, d=average_sampling_interval)
        
        # Calculate amplitudes and phases
        amplitude = np.abs(fft_result)*2/n
        phase = np.angle(fft_result)

        # Save FFT results in Code
        fft_results[product_flag] = (fft_freq, fft_result, amplitude, phase)

        # Create a DataFrame for the FFT results
        fft_df = pd.DataFrame({
            'Frequency (Cycles/Day)': fft_results[product_flag][0] * 86400,
            'Amplitude': np.abs(fft_results[product_flag][1])*2/n,
            'Phase(Radians)': np.angle(fft_results[product_flag][1])
        })
        
        # Save the DataFrame to a text file (CSV format)
        fft_df.to_csv(f'fft_{product_flag}_results.csv', index=False)
        print(f"FFT results saved for product_flag {product_flag}")

        # Plot FFT results (optional)
        plt.figure(figsize=(14, 10))
        plt.plot(fft_freq*86400, np.abs(fft_result)*2/n, 'or', label='FFT')  # FFT plot; 5400 seconds is about the time of one revolution
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
        plt.figure()
        plt.plot(fft_freq*86400, phase, 'or')
        plt.xscale('log')
        plt.xlabel('Frequency (cycles/day)')
        plt.ylabel('Phase Angle')
        plt.title(f'Frequency Versus Phase {product_flag}')
        plt.grid(True)
        plt.legend()

        # #Plot Data
        plt.figure()
        plt.plot(time_vector, data, label=f'Original Data for {product_flag} GRACE-FO C: 2024-05-01')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature (Celsius)')
        plt.title(f'Original Data for {product_flag}')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        print(f"Plots generated for product_flag {product_flag}")
        
    return fft_results



def analyze_files(vndrwl_files, fuel_est_files, marker, product_flag_index, data_indices_vndrwl, data_indices_fuel, time_s_index, time_ms_index):
    
    #initialize nested data structure and this function defininition will populate the data structure as the files are read
    nested_data={}

    def process_vndrwl_file(filename):

        date=extract_date_from_filename
        print('Date Extracted: {date}')

        satellite_type=extract_satellite_type_from_filename(filename)
        print('Satellite Marker Extracted: {satellite_type}')

        if date not in nested_data[date]:
                nested_data[date]={}
        
        if satellite_type not in nested_data[date]:
            nested_data[date][satellite_type]= {
                'Temperature and Pressure': [],
                'Fuel Mass Estimates': []
            }
            
        #Initialize lists for data storage
        data_vectors = {                            
            'skin_temp': [],    #Zenith direction skin temperature (Celsius)
            'adap_temp': [],    #Nadir direction skin temperature (Celsius)
            'tank_press':[],    #Actual tank pressure in the high CGPS pressure part (Bar)
            'reg_press': [],    #Actual tank pressure in the low CGPS pressure part (Bar)
            'avg_temp':  [],    #Averages of Skin and Adap
            'avg_press': [],    #Averages of Tank and Reg
            'vndrwl_mass_est': [], #Van der Waals estimate of gas in tank
            'fuel_est':  []     #Fuel Mass estimates from other file structure
            }
        
        # Initialize empty time vector, should be the same for every product
        time_vector=[]

        #Rewrite the function to specifically process the Van-Der-Waals type file
        process_file_past_header(filename, marker, product_flag_index, data_indices_vndrwl, time_s_index, time_ms_index, data_vectors, time_vector):

        #Avergae the Pressure and Temperature Readings
        avg_press= np.mean(data_vectors['reg_press'])
        avg_temp= np.mean(data_vectors['skin_temp'])

        #Define Variables for Van Der Waals Calculation
        # Variable initiation to compute mass differences
        v = 52      # volume of fuel tanks in liters
        a = 1.370   # constant for N2 (bar L^2/ mol^2)
        b = 0.0387  # constant for N2 (L/mol)
        R = 0.08314 # ideal gas constant in (bar L/ mol K)
        n0 = 0.5     # initial guess for newton raphson
        molar_mass_N2 = 28.006148008 #molar mass of N2 in g/mol
        total_mass = 16       # 16 kg of mass per fuel tank 


        #Append the processed data
        nested_data[date][satellite_type]['Temperature and Pressure'].append({
            'filename': filename,
            data_vectors:{
                'temperature': data_vectors[('skin_temp'+'adap_temp')/2],
                'pressure': data_vectors[('(reg_press'+'tank_press')/2]
            }
        })
        
    #Function for Processing the Fuel Mass Estimate Files
    def process_fuel_est_file(filename):
        date=extract_date_from_filename(filename)
        satellite_type= extract_satellite_type_from_filename(filename)

        if date not in nested_data:
            nested_data[date]={}

        if satellite_type not in nested_data[date]:
            nested_data[date][satellite_type]={
                    "Temperature and Pressure": [],
                    'Fuel Mass Estimates': []
                }
            
        #Initialize list for fuel mass estimates
        fuel_mass_estimates = []

        #Process the fuel mass estimates file
        with open(filename, 'r') as file:
            for line in file:
                if marker in line:
                    continue
                        
                columns=line.split

                if len(columns) > max(product_flag_index, *data_indices_fuel, time_s_index, time_ms_index):
                    fuel_mass_est = float(columns[data_indices_fuel[0]])
                    nested_data[date][satellite_type]['Fuel Mass Estimates'].append(fuel_mass_est)
            
    for vndrwl_file in vndrwl_files:
        process_vndrwl_file(vndrwl_file)
    for fuel_est_file in fuel_est_files:
        process_fuel_est_file(fuel_est_file)


    return nested_data


#Define Variables for the Data Bypass
vndrwl_files=glob.glob(r'C:\data\Fuel Leak Data\TNK1A_*-*-*_*_04.txt') #Use glob to get all files that contain the data for Van-Der-Waals Calculations
fuel_est_files=glob.glob(r'C:\data\Fuel Leak Data\MAS1A_*-*-*_*_04.txt') #use glob to get all files that contain the fuel mass estimates the tanks of both twin stellitss per hour
marker='# End of YAML header'
product_flag_index= 6 
data_indices_vndrwl= [7,8,9,10]                       # Adjust this index if the product flag is in a different column
data_indices_fuel= [7,8]
time_s_index= 0                         
time_ms_index= 1

# Initialize empty lists for storing data
time_vector = []        # Initialize empty time vector, should be the same for every product

# Analyze the files
nested_data = analyze_files(vndrwl_files, fuel_est_files, marker, product_flag_index, data_indices_vndrwl, data_indices_fuel, time_s_index, time_ms_index)

# fft_results = perform_fft(data_vectors, time_vector)