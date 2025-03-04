import os
import csv
import time
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import glob
import datetime


years = list(range(2020, 2024))
issue = 'sexuality'
trial = 0
model_path = f"{issue}-relevance-roberta-large-{trial}"
max_length = 512

batch_size = 3200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

def get_predictions(texts):
    # Tokenize and encode the batch of texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predictions = probs.argmax(dim=1).tolist()  # List of predictions
    return predictions 

def relevance_filtering(file_):
    # Initialize missing lines count
    missing_lines_count = 0
    missing_records_file = 'missing_records.csv'

    # Check if the missing records file exists; if not, create it with a header
    if not os.path.exists(missing_records_file):
        with open(missing_records_file, 'w', newline='') as missing_file:
            missing_writer = csv.writer(missing_file)
            missing_writer.writerow(['Filename', 'MissingLinesCount', 'Timestamp'])

    print(f"Pruning file {file_}")
    start_time = time.time()
    
    output_file_path = file_.replace("lang", "rel")
    with open(file_, "r", encoding='utf-8-sig', errors='ignore') as input_file, \
         open(output_file_path, "w", encoding='utf-8-sig', errors='ignore', newline='') as output_file:
        reader = csv.reader((line.replace('\x00', '') for line in input_file))
        writer = csv.writer(output_file)

        batch_texts = []
        batch_lines = []
        total_lines = 0
        relevant_lines = []  # Collect relevant lines to write in bulk

        for id_, line in enumerate(reader):
            if id_ % 1000 == 0 and id_ > 0:
                print(f"Processed {id_} lines")
            if len(line) >= 3:
                text = line[2].strip().replace("\n", " ")
                batch_texts.append(text)
                batch_lines.append(line)
                total_lines += 1

                # If batch is full, process it
                if len(batch_texts) == batch_size:
                    predictions = get_predictions(batch_texts)
                    for idx, pred in enumerate(predictions):
                        if pred == 1:  # Relevant
                            relevant_lines.append(batch_lines[idx])
                    # Clear the batches
                    batch_texts.clear()
                    batch_lines.clear()
                    
                    # Write relevant lines to output file in bulk
                    if relevant_lines:
                        writer.writerows(relevant_lines)
                        relevant_lines.clear()
            else:
                print(f"Skipping line {id_}: insufficient columns ({len(line)} found)")
                missing_lines_count += 1  # Increment missing lines count

        # Process any remaining texts in the last batch
        if batch_texts:
            predictions = get_predictions(batch_texts)
            for idx, pred in enumerate(predictions):
                if pred == 1:
                    relevant_lines.append(batch_lines[idx])
            # Write any remaining relevant lines
            if relevant_lines:
                writer.writerows(relevant_lines)
                relevant_lines.clear()

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Time taken to prune {file_}: {elapsed_minutes:.2f} minutes")
    print(f"Total lines processed: {total_lines}")

    # Record the missing lines count in the missing records CSV file
    with open(missing_records_file, 'a', newline='') as missing_file:
        missing_writer = csv.writer(missing_file)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        missing_writer.writerow([file_, missing_lines_count, timestamp])


data_folder = f"/raid/ryu64/Output/{issue}"

if __name__ == "__main__":
    for year in years:
        input_pattern = os.path.join(data_folder, f"RC_{year}-*__lang_filtered.csv")
        files = glob.glob(input_pattern)
        for file_ in files:
            relevance_filtering(file_)
