import os
import tarfile #to undo the zip
import fnmatch #for pattern matching
#Define Folder paths for data file extraction
input_folder ='C:\\data\\data'
output_folder_c='C:\\data\\AHK1AC'
output_folder_d='C:\\data\\AHK1AD'
#Check that output directories exist
os.makedirs(output_folder_c, exist_ok=True)
os.makedirs(output_folder_d, exist_ok=True)

# Function to match the specific file name patterns
def match_file_pattern(filename, satellite_type):
    # Pattern example: AHK1A_2024-05-01_C_04.txt or AHK1A_2024-05-01_D_04.txt
    pattern = f"AHK1A_????-??-??_{satellite_type}_04.txt"
    return fnmatch.fnmatch(filename, pattern)

# Function to extract specific files from tgz and delete the tgz file
def extract_and_sort_tgz_files(input_folder, output_folder_c, output_folder_d):
    # Iterate over each file in the input folder
    for tgz_filename in os.listdir(input_folder):
        # Check if the file is a .tgz file
        if tgz_filename.endswith('.tgz'):
            tgz_path = os.path.join(input_folder, tgz_filename)
            
            try:
                # Open the .tgz file
                with tarfile.open(tgz_path, 'r:gz') as tar_ref:
                    # Get the list of files in the tgz
                    file_list = tar_ref.getnames()
                    
                    # Loop through the files in the tgz
                    for file in file_list:
                        # Check if the file matches the desired pattern ("C" or "D" satellite)
                        if match_file_pattern(file, 'C'):
                            # Extract to the C folder
                            tar_ref.extract(file, output_folder_c)
                            print(f'Extracted {file} to {output_folder_c}')
                        
                        elif match_file_pattern(file, 'D'):
                            # Extract to the D folder
                            tar_ref.extract(file, output_folder_d)
                            print(f'Extracted {file} to {output_folder_d}')
                
                # After extracting, delete the .tgz file
                os.remove(tgz_path)
                print(f'Deleted {tgz_filename}')
                
            except EOFError:
                print(f"Error: {tgz_filename} is corrupted. Skipping file.")
            
            except Exception as e:
                print(f"Unexpected error processing {tgz_filename}: {str(e)}")

# Run the function
extract_and_sort_tgz_files(input_folder, output_folder_c, output_folder_d)

print("Extraction and sorting complete!")