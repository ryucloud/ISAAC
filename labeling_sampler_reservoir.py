import csv
import random
import os
from math import floor
import sys

csv.field_size_limit(sys.maxsize)

### Hyper-parameters
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

# Directory containing CSV files
dir_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "Output", issue, "lang_filtered"
)

def get_unique_keywords(keyword_str, max_keywords=100):
    """
    Extract and count unique keywords from the input string.
    
    Args:
        keyword_str (str): Comma-separated keyword string
        max_keywords (int): Maximum number of keywords to process
    
    Returns:
        list: List of unique keywords
        int: Number of unique keywords
    """
    try:
        # Split by comma and clean each keyword
        keywords = keyword_str.replace('\t', ',').split(',')
        
        # Use a set for uniqueness
        cleaned_keywords = set()
        for kw in keywords:
            kw = kw.strip()
            # Example: 'fat:' or 'thin:' special logic, if needed
            if 'fat:' in kw or 'thin:' in kw:
                parts = kw.split(':')
                if len(parts) > 1:
                    cleaned_keywords.add(f"{parts[0].strip()}: {parts[1].strip()}")
            elif kw:
                cleaned_keywords.add(kw)
        
        unique_keywords = list(cleaned_keywords)[:max_keywords]
        return unique_keywords, len(unique_keywords)
    except Exception as e:
        print(f"Error processing keywords: {e}")
        return [], 0

# Calculate how many samples to take per year, per category (top/bottom/random)
total_samples_per_year = sample_size // len(years)
samples_per_type_per_year = total_samples_per_year // 3

# Dictionary to store final samples for each annotator
all_samples = {
    0: [],  # annotator 0
    1: []   # annotator 1
}

# Process each year
for year_no, year in enumerate(years):
    print(f"\nProcessing year {year}...")
    
    dir_list = sorted(os.listdir(dir_path))

    # Reservoirs (lists) for top, bottom, random
    top_reservoir = []
    bottom_reservoir = []
    random_reservoir = []

    total_docs = 0  # How many docs processed for this year

    # Iterate through each file in the directory
    for file in dir_list:
        if str(year) in file and "lang_filtered.csv" in file:
            print(f"Processing file: {file}")
            try:
                with open(os.path.join(dir_path, file), "r", encoding='utf-8-sig', errors='ignore') as input_file:
                    reader = csv.reader(x.replace('\0', '') for x in input_file)
                    for id_, line in enumerate(reader):
                        # Skip the header row
                        if id_ == 0:
                            continue
                        try:
                            # Basic row validation: must have at least 3 columns for text
                            if line and len(line) > 2 and line[2].strip():
                                # Extract original_id from first column
                                original_id = line[0].strip()

                                text = line[2].strip().replace("\n", " ")
                                
                                # If there's a keywords column (index 7), parse it
                                if len(line) > 7:
                                    keywords, unique_count = get_unique_keywords(line[7])
                                else:
                                    keywords, unique_count = [], 0
                                
                                total_docs += 1

                                # ===========================
                                #    TOP SAMPLES (max)
                                # ===========================
                                if len(top_reservoir) < samples_per_type_per_year:
                                    # Not yet filled the top reservoir
                                    top_reservoir.append((unique_count, text, keywords, file, original_id))
                                    # Keep it sorted in descending order of unique_count
                                    top_reservoir.sort(key=lambda x: x[0], reverse=True)
                                else:
                                    # If this doc has more unique keywords than the last in top_reservoir
                                    if unique_count > top_reservoir[-1][0]:
                                        top_reservoir[-1] = (unique_count, text, keywords, file, original_id)
                                        top_reservoir.sort(key=lambda x: x[0], reverse=True)

                                # ===========================
                                #    BOTTOM SAMPLES (min)
                                # ===========================
                                if len(bottom_reservoir) < samples_per_type_per_year:
                                    bottom_reservoir.append((unique_count, text, keywords, file, original_id))
                                    bottom_reservoir.sort(key=lambda x: x[0])
                                else:
                                    if unique_count < bottom_reservoir[-1][0]:
                                        bottom_reservoir[-1] = (unique_count, text, keywords, file, original_id)
                                        bottom_reservoir.sort(key=lambda x: x[0])

                                # ===========================
                                #    RANDOM SAMPLES
                                # ===========================
                                # Using reservoir sampling
                                if len(random_reservoir) < samples_per_type_per_year:
                                    random_reservoir.append((unique_count, text, keywords, file, original_id))
                                else:
                                    s = random.randint(0, total_docs - 1)
                                    if s < samples_per_type_per_year:
                                        random_reservoir[s] = (unique_count, text, keywords, file, original_id)
                        except Exception as e:
                            print(f"Error processing line {id_} in {file}: {e}")
                            continue
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

    print(f"{total_docs} documents processed for year {year}.")

    # Combine the top, bottom, and random samples
    year_samples = top_reservoir + bottom_reservoir + random_reservoir

    # Assign sample type labels in the same order
    sample_types = (
        ["top_sample"] * len(top_reservoir) +
        ["bottom_sample"] * len(bottom_reservoir) +
        ["random_sample"] * len(random_reservoir)
    )

    # Assign random IDs, store results for the year
    random_ids_used = set()
    year_sample_data = []
    
    for i, (unique_count, text, keywords, file, original_id) in enumerate(year_samples):
        rand_int = random.randint(100000, 999999)
        while rand_int in random_ids_used:
            rand_int = random.randint(100000, 999999)
        random_ids_used.add(rand_int)
        
        year_sample_data.append({
            'random_id': rand_int,
            'text': text,
            'keywords': keywords,
            'file': file,
            'sample_type': sample_types[i],
            'original_id': original_id
        })

    # Append these samples to each annotator
    for annot in range(num_annot):
        # all_samples[annot] is a list of dict
        all_samples[annot].extend(year_sample_data)

# ========================================
# After processing all years, write output
# ========================================
for annot in range(num_annot):
    sample_file_path = os.path.join(dir_path, f"relevance_sample_{issue}_{annot}.csv")
    sample_key_file_path = os.path.join(dir_path, f"relevance_sample_key_{issue}_{annot}.csv")
    
    with open(sample_file_path, "w", encoding='utf-8', newline='') as sample_file, \
         open(sample_key_file_path, "w", encoding='utf-8', newline='') as sample_file_key:
        
        writer = csv.writer(sample_file)
        writer_key = csv.writer(sample_file_key)
        
        # Write headers
        writer.writerow([
            "random_id", "text"
        ])
        # Key file: random_id, file, original_id, keywords, sample_type
        writer_key.writerow(["random_id", "file", "original_id", "keywords", "sample_type"])
        
        # Shuffle samples before writing so we don't group them year by year
        random.shuffle(all_samples[annot])
        
        # Write rows
        for data in all_samples[annot]:
            writer.writerow([
                data['random_id'],
                data['text'],
                "",  # placeholders
                "",
                ""
            ])
            writer_key.writerow([
                data['random_id'],
                data['file'],
                data['original_id'],
                ",".join(data['keywords']),
                data['sample_type']
            ])

print("\nProcessing complete!")
print(f"Total samples per annotator: {len(all_samples[0])}")