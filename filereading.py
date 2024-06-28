input_filename=r'C:\data\AHK1A_2024-05-01_C_04.txt'
output_filename="Temps_2024_05_01_C.txt"
marker='# End of YAML header'
product_flag_index=5
product_column_index=7
time_s_index=0
time_ms_index=1

def check_product_flag(product_flag):

    # Define the sequences you are looking for
    sequences = {
        'tesu': '00111100000000000100000001000000', #bit 14 product flag, 
        'taicu':'00111100000000001000000001000000',  #bit 15 product flag; GF1 ACC Hk N ICU temps
        'tisu': '00111100000000010000000001000000',  #bit 16 product flag
        # Add more sequences as needed
    }

    for key, sequence in sequences.items():
        # If you find a match, exit the loop and return True
        if product_flag == sequence:
            return key  

    return None  # If no match is found, return None'

with open(input_filename, 'r') as input_file:
    with open(output_filename, 'w') as output_file:
        # Write header to the output file
        header = "Time_Seconds, Time_Microseconds, Time_Referenced, Product_Type, Interval_Since_Last_Same_Flag, Data\n"
        output_file.write(header)
        marker_found=False
        reference_s_ms_combined=None
        last_seen_times = {}  # Dictionary to store the last seen time for each product flag
        for line in input_file:
            if marker_found:
                columns= line.split()
            
                if len(columns) > max(product_flag_index, product_column_index, time_s_index, time_ms_index):
                    time_s = float(columns[time_s_index])
                    time_ms = float(columns[time_ms_index])
                    current_time_combined=time_s+time_ms/1000000
                        
                    if reference_s_ms_combined is None:
                        reference_s_ms_combined = current_time_combined

                    product_flag = columns[product_flag_index]
                    product_flag_key = check_product_flag(product_flag)
                    
                    if product_flag_key=='tisu':
                        product_data = columns[product_column_index]
                        time_referenced=current_time_combined-reference_s_ms_combined
                        # Calculate the interval since the last same product flag
                        if product_flag_key in last_seen_times:
                            interval_since_last = current_time_combined - last_seen_times[product_flag_key]
                        else:
                            interval_since_last = None  # No previous occurrence

                        # Update the last seen time for the current product flag
                        last_seen_times[product_flag_key] = current_time_combined

                        # Format the interval stringS
                        interval_str = f"{interval_since_last:.6f}" if interval_since_last is not None else "N/A"

                        # Convert and format the data as required
                        formatted_line = f"{time_s:.6f}, {time_ms:.6f}, {time_referenced:.6f}, {product_flag_key}, {interval_str}, {product_data}\n"
                        output_file.write(formatted_line)

            elif marker in line:
                marker_found = True
                print(f"Marker found: {marker}")
