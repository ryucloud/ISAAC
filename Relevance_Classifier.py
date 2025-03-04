# Import needed modules
# NOTE: Check HuggingFace's website for how to install PyTorch and transformers
# NOTE: sklearn is installed as scikit-learn
import torch
from transformers.file_utils import is_tf_available, is_torch_available
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import csv
import random
from sklearn import utils
import pandas as pd
from scipy import stats as st
import sys
import math
from collections import Counter
import pathlib

# data type
post_type = "comments"
years = list(range(2011,2013)) # for the full range, use: list(range(2007,2017))

# issue
issues = ['sexuality', 'age', 'skintone', 'race', 'ability', 'weight']
dimensions = {'sexuality': ['gay', 'straight'], 'age': ['young', 'old'], 'skintone': ['dark', 'white'],
              'race': ['black', 'white'], 'ability': ['abled', 'disabled'], 'weight': ['fat', 'thin']}
issue = "race"

# NOTE: If training a new model, set the variable below to True. If loading from disk, set it to False
trial = 0
training = False
retraining = False

# Set random seed for repeatable results in subsequent runs
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

set_seed(1)

# choose the model to use
# NOTE: BERT from 2018 is a basic, but good choice for many use cases
model_name = "roberta-large"
# Number of tokens allowed in a single document
# NOTE: Tokens further than 512 are unlikely to change whether a text is relevant or not
max_length = 512

# Create the object that breaks docs into BERT tokens
tokenizer = RobertaTokenizerFast.from_pretrained(model_name,do_lower_case=True)

# Reads the CSV corpus data
texts = {} 
ratings = {0:{},1:{}}
for rater in range(2):
    with open("Samples/relevance_sample_{}_{}_rated.csv".format(rater,issue),"r", encoding='utf-8',errors='ignore') as f:
        reader = csv.reader(f)
        for idx,line in enumerate(reader):
            if idx != 0 and len(line) > 0:
                try:
                    if rater == 0:
                        texts[int(line[0].strip())] = line[1].strip()
                    # if line[2].strip() != "-1":
                    ratings[rater][int(line[0].strip())] = int(line[2].strip())
                    # else:
                    #     ratings[rater][int(line[0].strip())] = 0
                except:
                    print(rater)
                    print(idx)
                    print(line)
                    break

labels = {}
for id_ in ratings[0]:
    if ratings[0][id_] == ratings[1][id_]:
        if ratings[0][id_] == -1:
            labels[id_] = 0
        else:
            labels[id_] = ratings[0][id_]
    elif ratings[0][id_] == 1 or ratings[1][id_] == 1:
        labels[id_] = 1
    else:
        labels[id_] = 0

texts = list(texts.values())
final_labels = list(labels.values())

print(" Number of annotated comments: {}".format(len(texts)))
print(Counter(final_labels))

def dataset_split(texts,labels,proportion):
    training_id = random.sample(range(len(texts)),math.floor(proportion*len(texts)))
    test_id = [i for i in range(len(texts)) if i not in training_id]
    training_texts = []
    training_labels = []
    test_texts = []
    test_labels = []
    for idx,i in enumerate(texts):
        if idx in training_id:
            training_texts.append(i)
            training_labels.append(labels[idx])
        elif idx in test_id:
            test_texts.append(i)
            test_labels.append(labels[idx])
        else:
            raise Exception
    
    return training_texts,test_texts,training_labels,test_labels

# NOTE: I'm using the very helpful sklearn function for breaking texts and labels into shuffled train and test sets
if not pathlib.Path("train_texts.csv").is_file():

    print("creating training, validation and test sets (80/10/10 split)")

    train_texts, valid_texts_init, train_labels, valid_labels_init = dataset_split(texts,final_labels,proportion=0.8)
    valid_texts, test_texts, valid_labels, test_labels = dataset_split(valid_texts_init,valid_labels_init,proportion=0.5)
    
    with open("train_texts_rel_{}.csv".format(issue),"w",encoding='utf-8',errors='ignore',newline="") as f:
        writer = csv.writer(f)
        for i in train_texts:
            writer.writerow([i])
    
    with open("train_labels_rel_{}.txt".format(issue),"w",encoding='utf-8',errors='ignore') as f:
        for i in train_labels:
            print(i,file=f)

    with open("valid_texts_rel_{}.csv".format(issue),"w",encoding='utf-8',errors='ignore',newline="") as f:
        writer = csv.writer(f)
        for i in valid_texts:
            writer.writerow([i])
    
    with open("valid_labels_rel_{}.txt".format(issue),"w",encoding='utf-8',errors='ignore') as f:
        for i in valid_labels:
            print(i,file=f)

    with open("test_texts_rel_{}.csv".format(issue),"w",encoding='utf-8',errors='ignore',newline="") as f:
        writer = csv.writer(f)
        for i in test_texts:
            writer.writerow([i])
    
    with open("test_labels_rel_{}.txt".format(issue),"w",encoding='utf-8',errors='ignore') as f:
        for i in test_labels:
            print(i,file=f)

else:

    print("loading training, validation and test sets (80/10/10 split)")

    train_texts = []
    with open("train_texts_rel_{}.csv".format(issue),"r",encoding='utf-8',errors='ignore') as f:
        reader = csv.reader(f)
        for i in reader:
            train_texts.append(i[0])

    train_labels = []
    with open("train_labels_rel_{}.txt".format(issue),"r",encoding='utf-8',errors='ignore') as f:
        for i in f:
            train_labels.append(int(i.strip()))

    valid_texts = []
    with open("valid_texts_rel_{}.csv".format(issue),"r",encoding='utf-8',errors='ignore') as f:
        reader = csv.reader(f)
        for i in reader:
            valid_texts.append(i[0])

    valid_labels = []
    with open("valid_labels_rel_{}.txt".format(issue),"r",encoding='utf-8',errors='ignore') as f:
        for i in f:
            valid_labels.append(int(i.strip()))

    test_texts = []
    with open("test_texts_rel_{}.csv".format(issue),"r",encoding='utf-8',errors='ignore') as f:
        reader = csv.reader(f)
        for i in reader:
            test_texts.append(i[0])

    test_labels = []
    with open("test_labels_rel_{}.txt".format(issue),"r",encoding='utf-8',errors='ignore') as f:
        for i in f:
            test_labels.append(int(i.strip()))

# NOTE: The transformers library requires labels to be in LongTensor, not Int format
train_labels=torch.from_numpy(np.array(train_labels)).type(torch.LongTensor)#data type is long
valid_labels=torch.from_numpy(np.array(valid_labels)).type(torch.LongTensor)#data type is long
test_labels=torch.from_numpy(np.array(test_labels)).type(torch.LongTensor)#data type is long

print("Number of training documents: {}".format(len(train_texts)))
print("Number of validation documents: {}".format(len(valid_texts)))
print("Number of test documents: {}".format(len(test_texts)))

weights = list(utils.compute_class_weight('balanced', classes=np.array(np.unique(train_labels)), y=np.array(train_labels)))
print(type(weights))
print("weights")
print(weights)

# Tokenize the training and validation data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

# Transformers requires custom datasets to be transformed into PyTorch datasets before training. The following function makes the transition
class relevance_data(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = relevance_data(train_encodings, train_labels)
valid_dataset = relevance_data(valid_encodings, valid_labels)
test_dataset = relevance_data(test_encodings, test_labels)

# load the model
# NOTE: If we were using GPUs, this would be where CUDA would be invoked

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=400,               # log & save weights each logging_steps
    save_steps=400,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([float(i) for i in weights]).to(device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Based on the value of [Training], the next piece of code either loads a fine-tuned model from disk or trains a new one
if training:

    # Create the trainer object with variables defined so far
    trainer = CustomTrainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
    )

    # Train the model
    trainer.train()

    # saving the fine-tuned model & tokenizer
    model_path = "sexuality-relevance-roberta-large-{}".format(trial)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

else: # if loading from disk
    model_path = "sexuality-relevance-roberta-large-{}".format(trial)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    if retraining:

        print("Retraining for 1 epoch.")

        trainer = CustomTrainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=valid_dataset,          # evaluation dataset
        )

        # Train the model
        trainer.train()

        # saving the fine-tuned model & tokenizer
        model_path = "sexuality-relevance-roberta-large-{}".format(trial)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

# Get the labels for each of the evaluation set documents
def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return [0,1][probs.argmax()] # 0 is irrelevant, 1 is relevant


# Evaluation

# TODO: Must be performed on the test data (the remainder of the corpus), NOT on the evaluation set used during training

# NOTE: Since the sklearn functions for precision, recall and f1 did not work properly, I wrote a short script below that calculates them from scratch
tp = 0
tn = 0
fp = 0
fn = 0

predictions = []
counter = 0
for text in test_texts:
    predictions.append(get_prediction(text))
    counter += 1

    if counter % 500 == 0:
        print(counter)

for idx,prediction in enumerate(predictions):
    
        if test_labels[idx] == 0:
            if prediction == 0:
                tn += 1
            elif prediction == 1:
                fp += 1
            else:
                raise Exception
        elif test_labels[idx] == 1:
            if prediction == 0:
                fn += 1
            elif prediction == 1:
                tp += 1
            else:
                raise Exception

precision = float(tp) / float(tp + fp)
print("Precision: {}".format(precision))
recall = float(tp) / float(tp + fn)
print("Recall: {}".format(recall))
F1 = 2 * float(precision * recall) / float(precision + recall)
print("F1: {}".format(F1))