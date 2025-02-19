import csv
import random
import os
from math import floor

num_annot = 2
sample_size = 1500  
post_type = "comments"  
years = list(range(2007, 2024)) 
issue = 'ability'
dimensions = {
    'sexuality': ['gay', 'straight'],
    'age': ['young', 'old'],
    'skintone': ['dark', 'white'],
    'race': ['black', 'white'],
    'ability': ['abled', 'disabled'],
    'weight': ['fat', 'thin']
}

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Output/{issue}/lang_filtered/")

final_sample = {}
random_ids = {}
sample_tag = {}

for year_no, year in enumerate(years):
    allocation = int(sample_size / len(years))  # Allocation per year
    idx = 0  # Global sample index

    dir_list = sorted(os.listdir(dir_path))

    # Dictionaries to store sample information
    lengths = {}
    keyword_ids = {}
    time = {}
    sample_data = {}

    for file in dir_list:
        if str(year) in file and "lang_filtered.csv" in file:
            print(f"Processing file: {file}")
            file_path = os.path.join(dir_path, file)
            try:
                with open(file_path, "r", encoding='utf-8-sig', errors='ignore') as input_file:
                    reader = csv.reader(x.replace('\0', '') for x in input_file)
                    try:
                        headers = next(reader)
                        text_index = 0  # Assuming 'text' is in the first column
                        keywords_index = 1  # Assuming 'keywords' are in the second column
                    except StopIteration:
                        print(f"Skipping empty file: {file}")
                        continue

                    for line in reader:
                        if len(line) <= max(text_index, keywords_index):
                            continue
                        text = line[text_index].strip()
                        if text == "":
                            continue

                        keywords_list = line[keywords_index].split(",")
                        for keyword in keywords_list:
                            keyword = keyword.strip()
                            if keyword not in keyword_ids:
                                keyword_ids[keyword] = [idx]
                            else:
                                keyword_ids[keyword].append(idx)

                        lengths[idx] = len(keywords_list)
                        # Store minimal sample data
                        sample_data[idx] = {
                            'file': file,
                            'text': text.replace("\n", " "),
                            'keywords': keywords_list
                        }
                        time[idx] = file
                        idx += 1

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

    print(f"{len(sample_data)} documents for year {year}")

    if len(lengths) == 0:
        print(f"No samples found for year {year}. Skipping to next year.")
        continue

    # Select top samples
    sorted_keywords = sorted(lengths.items(), key=lambda x: x[1], reverse=True)
    top_sample = []
    for i in sorted_keywords:
        top_sample.append(i[0])
        if len(top_sample) >= floor(allocation / 3):
            break

    # Select bottom samples
    bottom_sample = []
    while len(bottom_sample) < floor(allocation / 3):
        for keyword in keyword_ids:
            rel_lengths = {}
            for id in keyword_ids[keyword]:
                if id not in rel_lengths:
                    rel_lengths[id] = lengths[id]
            sorted_keywords_bottom = sorted(rel_lengths.items(), key=lambda x: x[1])
            for i in sorted_keywords_bottom:
                if i[0] in bottom_sample or i[0] in top_sample:
                    continue
                else:
                    bottom_sample.append(i[0])
                    break
            if len(bottom_sample) >= floor(allocation / 3):
                break

    # Select random samples
    remaining_ids = set(sample_data.keys()) - set(top_sample) - set(bottom_sample)
    random_sample_size = floor(allocation / 3)
    if len(remaining_ids) < random_sample_size:
        print(f"Not enough remaining samples to select random samples for year {year}.")
        random_sample = list(remaining_ids)
    else:
        random_sample = random.sample(list(remaining_ids), random_sample_size)

    final_sample[year] = top_sample + bottom_sample + random_sample

    # Assign sample types
    sample_types = (
        ["top_sample"] * len(top_sample) +
        ["bottom_sample"] * len(bottom_sample) +
        ["random_sample"] * len(random_sample)
    )
    for idx, sample_id in enumerate(final_sample[year]):
        sample_tag[sample_id] = sample_types[idx]

    # Generate random IDs
    for sample_id in final_sample[year]:
        rand_int = random.randint(100000, 999999)
        while rand_int in random_ids.values():
            rand_int = random.randint(100000, 999999)
        random_ids[sample_id] = rand_int

    # Write samples to files
    for annot in range(num_annot):
        sample_filename = os.path.join(dir_path, f"relevance_sample_{issue}_{annot}.csv")
        key_filename = os.path.join(dir_path, f"relevance_sample_key_{issue}_{annot}.csv")

        with open(sample_filename, "a+", encoding='utf-8', errors='ignore', newline='') as sample_file, \
             open(key_filename, "a+", encoding='utf-8', errors='ignore', newline='') as sample_key_file:

            writer = csv.writer(sample_file)
            writer_2 = csv.writer(sample_key_file)

            if year_no == 0 and annot == 0:
                writer.writerow([
                    "random_id", "text", "relevance_label",
                    f"attitude_label_{dimensions[issue][0]}",
                    f"attitude_label_{dimensions[issue][1]}"
                ])
                writer_2.writerow(["random_id", "file", "original_id", "keywords", "sample_type"])

            random.shuffle(final_sample[year])
            for sample_id in final_sample[year]:
                data = sample_data[sample_id]
                writer.writerow([random_ids[sample_id], data['text'], "", "", ""])
                writer_2.writerow([
                    random_ids[sample_id], data['file'], sample_id,
                    ",".join(data['keywords']), sample_tag[sample_id]
                ])

print("Sampling complete.")