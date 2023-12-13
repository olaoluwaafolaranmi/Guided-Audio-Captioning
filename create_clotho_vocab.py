import json
import os
from re import sub
from pathlib import Path
import csv
from itertools import chain
from tools.file_io import write_to_json


def create_vocab(dataset):

    train_path = f'data/{dataset}/csv_files/train.csv'
    test_path = f'data/{dataset}/csv_files/test.csv'
    val_path = f'data/{dataset}/csv_files/val.csv'
    save_path = f"data\{dataset}\vocab\word_list.json"

    dev_csv = load_csv(train_path)
    val_csv = load_csv(val_path)
    eval_csv = load_csv(test_path)   
    
    words = []
    field_caption = 'caption_{}'
    for item in chain(dev_csv, val_csv, eval_csv):

        for cap_ind in range(1, 6):
            sentence = item[field_caption.format(cap_ind)].lower()
            # remove fogotten space before punctuation and double space
            sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')
            sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
            sentence = '<sos> {} <eos>'.format(sentence).strip().split()
            words.extend(sentence)
    words.append('<ukn>')

    words = list(set(words))

    write_to_json(words,'vocabulary',save_path)
    
    return 


def load_csv(file_name):
    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj


if __name__ == '__main__':

    dataset = 'Clotho'
    create_vocab(dataset)

