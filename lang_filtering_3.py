import fasttext
import os
import csv
import time
import sys

# Increase the field size limit to handle larger fields
csv.field_size_limit(sys.maxsize) 

post_type = "comments"
years = [2023]  # Adjust the year range as necessary
dir_path = os.path.dirname(os.path.realpath(__file__))

# Load the fastText language identification model
model = fasttext.load_model('lid.176.bin')

def detect_language(text):
    predictions = model.predict(text)
    # The language code is returned with a prefix "__label__", which we remove
    return predictions[0][0].replace('__label__', '') 

for year_no, year in enumerate(years):
    print(year)
    for file in os.listdir(dir_path):
        if str(year) in file and "_output.csv" in file:
            print(file)
            with open(file, "r", encoding='utf-8', errors='ignore') as input_file, open("{}_lang_filtered.csv".format(file.split("output.csv")[0]),"w",encoding='utf-8',errors='ignore',newline='') as output_file:
                start_time = time.time()
                # Replace NUL characters with empty strings before passing to csv.reader
                reader = csv.reader((line.replace('\0', '') for line in input_file))
                writer = csv.writer(output_file)
                for id_, line in enumerate(reader):
                    try:
                        if id_ != 0:
                            if id_ % 10000 == 0:
                                print(id_)
                            text = line[2].strip().replace("\n", " ")
                            if detect_language(text) == 'en':
                                writer.writerow(line)
                    except IndexError as e:
                        print(f"Skipping line {id_ + 1} due to error: {e}")
                        continue  # Skip this line and continue with the next one
                end_time = time.time()
                print(f"Time taken to process {file}: {(end_time - start_time)/60} minutes")
