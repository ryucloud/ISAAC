import os
import csv
import glob

# Define the header
header = ["id", "parent id", "text", "author", "time", "subreddit", "score", "matched keywords"]

# Directory where your files are located
dir_path = os.path.dirname(os.path.realpath(__file__))

# Find all files matching the pattern
files_to_check = glob.glob(os.path.join(dir_path, "RC_*__lang_filtered.csv"))

for file_path in files_to_check:
    file_name = os.path.basename(file_path)
    
    # Open the file in read+write mode and check if it already has a header
    with open(file_path, 'r+', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        first_row = next(reader, None)
        
        # If the first row is not the header, add the header
        if first_row != header:
            # Save the rest of the file
            rest_of_file = csvfile.readlines()
            
            # Move the file pointer to the beginning of the file
            csvfile.seek(0)
            csvfile.truncate()  # Clear the file content
            
            # Write the header and the rest of the original content
            writer = csv.writer(csvfile)
            writer.writerow(header)
            if first_row:  # Only write the first row if it's not None
                writer.writerow(first_row)
            csvfile.writelines(rest_of_file)
            print(f"Header added to {file_name}")
        else:
            print(f"Header already present in {file_name}")
