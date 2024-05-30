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
            if marker_found:        #snippet to get all the data past the marker
                process_line(line)
            elif marker in line:
                marker_found=True
                continue

def process_line(line):
    print(line.strip())

#Define Variables for the Data Bypass
filename=r'C:\data\ACC1A_2020-07-01_C_04.txt' #the r with the prefix denotes the absolute path of the file
marker='# End of YAML header'

#Call the function
process_file_past_header(filename,marker)
