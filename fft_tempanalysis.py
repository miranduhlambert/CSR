#Level-1A Data Level-1A data is the result of “non-destructive” processing applied to Level-0 data. Sensor calibration factors are applied to convert the binary encoded measurements to engineering units. Where necessary,
#time tag integer second ambiguity is resolved and data is time tagged to the respective satellite receiver clock time.
#Editing and quality control flags are added, and the data is reformatted for further processing. This is considered
#“non-destructive” processing, meaning that the Level-1A data is reversible to Level-0, except for bad data packets.
#This level also includes the ancillary data products needed for processing to the next data level.

#Imports

import matplotlib.pyplot as plt

#Function for filtering datato Bypass Yaml Header marker

def process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors):

    marker_found= False

    with open(filename,'r') as file:

        for line in file:

            # Debugging output to trace the lines being read

            # print(f"Reading line: {line.strip()}")

            if marker_found: # snippet to get all the data part the marker 
        
                columns = line.split()

                # Ensure the line has at least 5 columns to avoid index errors
                
                if len(columns) > max(product_flag_index, product_column_index):
        
                    product_flag = columns[product_flag_index]  # Extract the fifth column
                    
                    # Debugging output to trace the product flag
   
                    #print(f"Product flag found: {product_flag}")

                    #extract data 
                    if product_flag in data_vectors:
                            product_data = columns[product_column_index]
                            data_vectors[product_flag].append(float(product_data))


            elif marker in line:
                marker_found=True
                # print(f"Marker found: {marker}")
   

def process_line(line):
    print(line.strip())

def check_product_flag(product_flag, data_vectors, line):

    # Define the sequences you are looking for
    sequences = {
        'tesu': '00111100000000000100000001000000 ', #bit 14 product flag
        'taicu': '00111100000000001000000001000000',  #bit 15 product flag
        'tisu': '00111100000000010000000001000000',  #bit 16 product flag
        'tcicu': '00111100000100000000000001000000'   #bit 20 product flag
        # Add more sequences as needed
    }

    for key, sequence in sequences.items():
        if product_flag == sequence:
            data_vectors[key].append(line.strip())
            return True  # If you find a match, exit the loop and return True

    return False  # If no match is found, return False'

    # Debugging output to trace the product flag check

    # print(f"Checking product flag: {product_flag} against {desired_sequence}")
    # return product_flag == desired_sequence

# Function to process and print the line for debugging

# def process_line(line):
#     print(f"Processing line: {line.strip()}")

def plot_data_vectors(data_vectors):
    for product_flag, data in data_vectors.items():
        if not data:
            continue
        #Plot the data
        plt.figure()
        plt.plot(range(len(data)), data, label=product_flag)
        plt.xlabel('Time of Day (hours)')
        plt.ylabel('Temperature (Celsius)')
        plt.title(f'Product Data Analysis for Product Flag: {product_flag}')
        plt.legend()
        plt.show()

#Define Variables for the Data Bypass

filename=r'C:\data\AHK1A_2020-07-01_C_04.txt' #the r with the prefix denotes the absolute path of the file
marker='# End of YAML header'
product_flag_index= 5                        # Adjust this index if the product flag is in a different column
product_column_index= 7                      # Adjust this index to the column where product data is located


# Initialize empty lists for storing data
data_vectors = {                            
    'tesu': [],
    'taicu': [],
    'tisu': [],
    'tcicu': []
    
    # Add more product flags as needed
    
    }


#Call the function

process_file_past_header(filename, marker, product_flag_index, product_column_index, data_vectors)

# Plot the data

plot_data_vectors(data_vectors)


