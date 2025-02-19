import ahocorasick
import os
import csv
import json
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import zstandard
import io
from os import cpu_count

# Helper function to load terms from a file
def load_terms(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# Load terms for Aho-Corasick automaton
dir_path = os.path.dirname(os.path.realpath(__file__))
fat_terms_file = os.path.join(dir_path, "weight_binary_regex_fat_SH.txt")
thin_terms_file = os.path.join(dir_path, "weight_binary_regex_thin_SH.txt")

fat_terms = load_terms(fat_terms_file)
thin_terms = load_terms(thin_terms_file)

# Initialize the Aho-Corasick automaton
automaton = ahocorasick.Automaton()

# Add terms with their categories
for term in fat_terms:
    automaton.add_word(term, ("fat", term))
for term in thin_terms:
    automaton.add_word(term, ("thin", term))

# Build the Aho-Corasick trie
automaton.make_automaton()

# File paths and configuration
post_type = "comments"
years = list(range(2020, 2024))
issue = "weight"
csv_header = ["id", "parent id", "text", "author", "time", "subreddit", "score", "matched patterns"]
dir_path = os.path.dirname(os.path.realpath(__file__))
full_path = os.path.join(dir_path, "Reddit", post_type, "")
issue_path = os.path.join(dir_path, "Output", issue)
os.makedirs(issue_path, exist_ok=True)

processed_files = set(f.split('_output.csv')[0] for f in os.listdir(issue_path) if f.endswith('_output.csv'))

# Process a single file
def process_single_file(file):
    file_path = os.path.join(full_path, file)
    output_csv_file = os.path.join(issue_path, f"{file.split('.zst')[0]}_output.csv")
    buffer = []
    buffer_size = 20
    total_lines = 0
    matched_lines = 0
    start_time = time.time()

    try:
        with open(file_path, 'rb') as fh, \
             open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_header)
            dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
            stream_reader = dctx.stream_reader(fh, read_across_frames=True)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

            for line in text_stream:
                total_lines += 1
                contents = json.loads(line)
                comment_text = contents['body'].strip().lower()

                matches = []
                for end_index, (category, term) in automaton.iter(comment_text):
                    matches.append(f"{category}: {term}")

                if matches:
                    matched_lines += 1
                    buffer.append([
                        contents.get("id", ""),
                        contents.get("parent_id", ""),
                        comment_text,
                        contents.get("author", ""),
                        datetime.fromtimestamp(int(contents.get("created_utc", 0))).strftime('%Y-%m-%d %H:%M:%S'),
                        contents.get("subreddit", ""),
                        contents.get("score", ""),
                        ', '.join(matches)
                    ])

                    if len(buffer) >= buffer_size:
                        writer.writerows(buffer)
                        buffer.clear()

            if buffer:
                writer.writerows(buffer)

    except Exception as e:
        print(f"Error processing file {file}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    print(f"Processed {file} in {elapsed_time:.2f} minutes")
    return total_lines, matched_lines

# Process all files for a specific month
def process_month(year, month, files):
    """Process all files for a specific month."""
    print(f"Start processing month: {year}-{month}")
    start_time = time.time()
    total_lines = 0
    matched_lines = 0

    for file in files:
        try:
            file_lines, file_matched = process_single_file(file)
            total_lines += file_lines
            matched_lines += file_matched
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    elapsed_time = (time.time() - start_time) / 60
    print(f"Completed processing month {year}-{month} in {elapsed_time:.2f} minutes")
    return total_lines, matched_lines

# Process files in parallel
def process_files_parallel():
    total_lines = 0
    matched_lines = 0

    max_workers = min(6, cpu_count())  # Use up to 8 processes or the number of available CPU cores
    print(f"Using {max_workers} processes for parallel processing.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for year in years:
            print(f"Processing year: {year}")
            start_year_time = time.time()

            # Group files by month
            files_by_month = {}
            for file in sorted(os.listdir(full_path)):
                if str(year) in file and file.endswith(".zst") and file.split('.zst')[0] not in processed_files:
                    month = file.split('-')[1]  # Assuming the format includes the month, e.g., "2013-01"
                    files_by_month.setdefault(month, []).append(file)

            # Submit each month as a separate task
            for month, files in sorted(files_by_month.items()):
                futures.append(executor.submit(process_month, year, month, files))

            # Collect results from all futures
            for future in futures:
                try:
                    month_lines, month_matched = future.result()
                    total_lines += month_lines
                    matched_lines += month_matched
                except Exception as e:
                    print(f"Error processing month: {e}")

            year_processing_time = (time.time() - start_year_time) / 60
            print(f"Completed processing year {year} in {year_processing_time:.2f} minutes")

    print(f"Total lines processed: {total_lines}")
    print(f"Total matched lines: {matched_lines}")

if __name__ == "__main__":
    start_time = time.time()
    process_files_parallel()
    print(f"Processing completed in {(time.time() - start_time) / 60:.2f} minutes")