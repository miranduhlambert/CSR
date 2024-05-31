#Level-1A Data Level-1A data is the result of “non-destructive” processing applied to Level-0 data. Sensor calibration factors are applied to convert the binary encoded measurements to engineering units. Where necessary,
#time tag integer second ambiguity is resolved and data is time tagged to the respective satellite receiver clock time.
#Editing and quality control flags are added, and the data is reformatted for further processing. This is considered
#“non-destructive” processing, meaning that the Level-1A data is reversible to Level-0, except for bad data packets.
#This level also includes the ancillary data products needed for processing to the next data level.



#Function for filtering datato Bypass Yaml Header marker
def process_file_past_header(filename,marker):

    marker_found= False

    with open(filename,'r') as file:

        for line in file:

            # Debugging output to trace the lines being read

            print(f"Reading line: {line.strip()}")

            if marker_found: # snippet to get all the data part the marker 
        
                columns = line.split()

                # Ensure the line has at least 5 columns to avoid index errors
                
                if len(columns) > product_flag_index:
        
                    product_flag = columns[product_flag_index]  # Extract the fifth column
                    
                    # Debugging output to trace the product flag

                    print(f"Product flag found: {product_flag}")
                    
                    product_flag = columns[product_flag_index]  # Extract the fifth column
                    
                    # Check if the product flag matches the desired sequence
                    if check_product_flag(product_flag):

                        process_line(line)

                        return # Exit after the first matching line

            elif marker in line:
                marker_found=True
                print(f"Marker found: {marker}")
   

def process_line(line):
    print(line.strip())

def check_product_flag(product_flag):

    # Define the sequence you are looking for
    desired_sequence = '00000100000000000000000000111111'

    # Debugging output to trace the product flag check
    print(f"Checking product flag: {product_flag} against {desired_sequence}")
    return product_flag == desired_sequence

# Function to process and print the line
def process_line(line):
    print(f"Processing line: {line.strip()}")

#Define Variables for the Data Bypass
filename=r'C:\data\ACC1A_2020-07-01_C_04.txt' #the r with the prefix denotes the absolute path of the file
marker='# End of YAML header'
product_flag_index= 5                        # Adjust this index if the product flag is in a different column


#Call the function
process_file_past_header(filename,marker)
