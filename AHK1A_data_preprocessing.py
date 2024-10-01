
import pandas as pd
import glob


#Function that checks the Product Flag Sequences:
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
    #If no match is found, return None
    return None

# Function to average two consecutive data points
def average_data(time1, time2, data1, data2):
    avg_time = (time1 + time2) / 2
    avg_values = (data1 + data2) / 2
    return avg_time, avg_values

#Define folder path where data files are stored
file_list = glob.glob(r'C:\data\AHK1AC\AHK1A_*-*-*_C_04.txt')

#Create a Variable to store the first time value
first_time=None

#Initialize varibales to track previous data
previous_data_type=None
previous_time= None
previous_value=None

#Initialize lists to store data
tesu_data=[]
taicu_data=[]
tisu_data=[]

#Loop over all text files with the specified file path/structure
for filename in file_list:
    #Defining variables for loop iteration:
    marker='# End of YAML header'
    marker_found=False

    #Specify Column Indices for data
    time_seconds_index= 0
    time_microseconds_index=1
    product_flag_index= 5
    product_column_index= 7

    print(f"Processing file: {filename}")

    #Open the file to read it
    with open(filename, 'r') as file:
        #Parse Lines in file 
        for line in file:
            #Until the marker is found
            if marker_found:
                #Remove all whitespaces from line and split it
                data=line.strip().split()
                #Use specificed indices to extract time data
                time_seconds=float(data[time_seconds_index])
                time_microseconds=float(data[time_microseconds_index])/1000000
                total_time=time_seconds+time_microseconds
                #Track the first time value
                if first_time is None:
                    first_time=total_time
                #Adjust the time to create a continuous time series starting at 0
                adjusted_time=total_time-first_time
                #Extract temperature data
                temperature=float(data[product_column_index])
                #Use specified indices to extract product flag
                product_flag=data[product_flag_index]
                #Call function to see if product_flag is a desired data type
                data_type=check_product_flag(product_flag)
                #If it's a valid data type
                if data_type:
                    current_value=float(data[product_column_index])
                    if previous_data_type==data_type:
                        #If current data type matches the previous, average the times and values
                       # If the current and previous flags match, average the data
                        avg_time, avg_values = average_data(previous_time, adjusted_time, previous_value, temperature)
                        
                        # Append averaged data to the corresponding list
                        if data_type == 'tesu':
                            tesu_data.append([data_type, avg_time, avg_values])
                        elif data_type == 'taicu':
                            taicu_data.append([data_type, avg_time, avg_values])
                        elif data_type == 'tisu':
                            tisu_data.append([data_type, avg_time, avg_values])
                    else:
                        previous_data_type=data_type
                        previous_time=adjusted_time
                        previous_value=temperature
                    
                #If the function returns a type, append it
            #Check for the marker in the line
            if marker in line:
                #Set marker_found to True where the marker line is found
                marker_found = True

# Convert lists into DataFrames
tesu_df = pd.DataFrame(tesu_data, columns=["Data Type", "Adjusted Time", "Temperature Values"])
taicu_df = pd.DataFrame(taicu_data, columns=["Data Type", "Adjusted Time", "Temperature Values"])
tisu_df = pd.DataFrame(tisu_data, columns=["Data Type", "Adjusted Time", "Temperature Values"]) 

# Save DataFrames to CSV files
tesu_df.to_csv('tesu_data_avg.csv', index=False)
taicu_df.to_csv('taicu_data_avg.csv', index=False)
tisu_df.to_csv('tisu_data_avg.csv', index=False)

        
        

        

