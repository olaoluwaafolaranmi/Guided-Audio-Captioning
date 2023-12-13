#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

from pathlib import Path
import os
import csv
import pickle
import json

def write_csv_file(csv_obj, file_name):

    with open(file_name, 'w') as f:
        writer = csv.DictWriter(f, csv_obj[0].keys())
        writer.writeheader()
        writer.writerows(csv_obj)
    print(f'Write to {file_name} successfully.')


def load_csv_file(file_name):

    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj


def load_pickle_file(file_name):

    with open(file_name, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj

def load_json_file(file_name):

    with open(file_name, 'rb') as f:
        data = json.load(f)
    return data


def write_pickle_file(obj, file_name):

    Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Write to {file_name} successfully.')


def write_to_json(obj, key , save_path):
    
    word_dict = {key : obj}
    
    json_str = json.dumps(word_dict, indent=3)
    with open(save_path, "w") as outfile:
        outfile.write(json_str)

