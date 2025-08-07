#!/usr/bin/env python3
"""
New pipeline for seizure detection.
This replaces the old approach of loading data from segments.
"""

import os
import time
from datetime import datetime
from catch22_feature_analysis import Catch22CumlClassifier
import pandas as pd
import yaml
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import joblib
import random
import sys
import logging
from log import OutputCapture, StdoutCapture, StderrCapture

# Import existing utilities
from new_dataloader import (
    load_epoch_data,
    read_all_events, 
    create_balanced_epochs,
    find_matching_config,
)
from new_analysis import analyze_classification
from mne_utils import setup_mne_for_processing
from detach_rocket.detach_classes import DetachRocket, DetachEnsemble
from sklearn.model_selection import StratifiedKFold, KFold

def load_dataset(config): #not used
    """Load the dataset."""

    # Configure MNE verbosity
    setup_mne_for_processing(verbose=config.get('mne_verbose', False))

    print("\n=== Preprocess Pipeline with Balanced Epochs ===")
    
    # Step 1: Read and label all events
    all_events, (seizure_subjects, bckg_subjects) = read_all_events(config)
    # train test split, using stratified k-fold cross validation
    split_seed = config.get('split_seed')
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    all_subjects = np.concatenate([seizure_subjects, bckg_subjects])
    all_labels = np.concatenate([np.ones(len(seizure_subjects)), np.zeros(len(bckg_subjects))])
    split_counter = 0
    for train_index, test_index in kf.split(all_subjects, all_labels):
        split_counter += 1
        train_subjects = all_subjects[train_index]
        test_subjects = all_subjects[test_index]
        print(f"Train subjects ids for split {split_counter}: {train_subjects}, train labels: {all_labels[train_index]}")
        print(f"Test subjects ids for split {split_counter}: {test_subjects}, test labels: {all_labels[test_index]}")

    # train_seizure_subjects, test_seizure_subjects = train_test_split(seizure_subjects, test_size=1-config['split_ratio'], random_state=split_seed)
    # train_bckg_subjects, test_bckg_subjects = train_test_split(bckg_subjects, test_size=1-config['split_ratio'], random_state=split_seed)
    # train_subjects = train_seizure_subjects + train_bckg_subjects
    # test_subjects = test_seizure_subjects + test_bckg_subjects
    # print(f"Train subjects: {len(train_subjects)}")
    # print(f"Test subjects: {len(test_subjects)}")
    # train events are all events from train subjects
        train_events = all_events[all_events['subject'].isin(train_subjects)]
    # test events are all events from test subjects
        test_events = all_events[all_events['subject'].isin(test_subjects)]

        if config.get('debug', False):
            print("\nReading and labeling events for split {}...".format(split_counter))
            if all_events.empty:
                print("No events found! Check your BIDS directory.")
                return
            print(f"Total events loaded for split {split_counter}: {len(all_events)}")
    
    # Step 3: Create balanced epochs
    
        print(f"Creating balanced epochs for train set for split {split_counter}...")
        train_epochs = create_balanced_epochs(train_events, config)
        print(f"Creating balanced epochs for test set for split {split_counter}...")
        test_epochs = create_balanced_epochs(test_events, config)


    print("\n=== Preprocess Pipeline Completed Successfully! ===")

    return train_epochs, test_epochs

def train_model(train_epochs, config, split_counter, cv=False):
    """Train the model."""
    global log_path
    # check if the config file exists
    if config.get('load_model', False) and not cv:
        matching_config = find_matching_config(config, config['model_dir'])
        if matching_config is not None:
            print(f"Found existing model. Loading model...")
            timestamp = matching_config['timestamp']
            data_file = os.path.join(config['model_dir'], f'{config["model_name"]}_{timestamp}.pkl')
            
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    model = joblib.load(f)
                print(f"Model loaded from: {data_file}")
                print(f"\n ===== Training Model Loaded ===== \n")
                log_path = os.path.join(config['model_dir'], f'{config["model_name"]}_{timestamp}_log.txt')
                # os.rename(os.path.join(config['model_dir'], 'temp.txt'), os.path.join(config['model_dir'], f'{config["model_name"]}_{timestamp}_log.txt'))
                return model
            else:
                print(f"Model file not found, training new model...")
        
    log_path = os.path.join(config['model_dir'], f'{config["model_name"]}_{config["timestamp"]}_split_{split_counter}_log.txt')

    X_train, y_train = load_epoch_data(train_epochs, config, split_name='train', split_counter=split_counter, cv=cv)
    
    print(f"\n ===== Training Model ===== \n")

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    if config['model_type'] == 'detach_rocket':
        model = DetachRocket('pytorch_minirocket', num_kernels=config['num_kernels'], verbose=False, device=device) # multivariate; input_shape=(n_samples, n_channels, timestamps)
    elif config['model_type'] == 'detach_ensemble':
        model = DetachEnsemble(num_models=config['num_models'], num_kernels=config['num_kernels'], model_type='pytorch_minirocket', verbose=False)
    elif config['model_type'] == 'catch22':
        model = Catch22CumlClassifier(batch_size=512, n_jobs=8)

    start_training_time = time.time()
    model.fit(X_train, y_train)
    end_training_time = time.time()
    print(f"Model training completed in: {end_training_time - start_training_time:.2f} seconds")
    
    # save model
    model_name = "{}_{}_split_{}".format(config['model_name'], config['timestamp'], split_counter)
    model_path = os.path.join(config['model_dir'], f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)
    print(f"Model saved to: {model_path}")

    # save model config to the same directory
    # TODO: we need configuration, number of epochs, performance(terminal output).
    config_path = os.path.join(config['model_dir'], f"{model_name}_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_path}")

    print(f"\n ===== Training Model Completed ===== \n")

    return model

def test_model(model, test_epochs, config, split_counter, cv=False):
    """Evaluate the model."""
    X_test, y_test = load_epoch_data(test_epochs, config, split_name='test', split_counter=split_counter, cv=cv)
    print(f"\n ===== Evaluating Model on test set ===== \n")

    # evaluate the model
    y_pred = model.predict(X_test)
    print(f"Test set size: {len(X_test)}")

    f1, precision, recall, accuracy, conf_mat, conf_mat_norm = analyze_classification(y_pred, y_test)

    print(f"\n ===== Evaluating Model Completed ===== \n")

    return y_pred


# Set up output capture at the top of your main file
log_path = 'temp.txt'
output_capture = OutputCapture(log_path, also_console=True)
sys.stdout = StdoutCapture(output_capture)
# sys.stderr = StderrCapture(output_capture)

if __name__ == "__main__":
    
    with open('./config_cv_c22.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config['timestamp'] = datetime.now().strftime("%m%d_%H%M%S") # add unique timestamp to the config
        config['dataset'] = config['bids_root'].split('/')[-1]

    os.makedirs(config['preprocessed_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)

    # train_epochs, test_epochs = load_dataset(config)
    # load data
    setup_mne_for_processing(verbose=config.get('mne_verbose', False))

    print("\n=== Preprocess Pipeline with Balanced Epochs ===")
    
    # Read and label all events
    all_events, (seizure_subjects, bckg_subjects) = read_all_events(config)
    # train test split, using stratified k-fold cross validation
    split_seed = config.get('split_seed')
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    all_subjects = np.concatenate([seizure_subjects, bckg_subjects])
    all_labels = np.concatenate([np.ones(len(seizure_subjects)), np.zeros(len(bckg_subjects))])
    split_counter = 0
    for train_index, test_index in kf.split(all_subjects, all_labels):
        split_counter += 1
        if split_counter == 1:
            continue
        train_subjects = all_subjects[train_index]
        test_subjects = all_subjects[test_index]
        print(f"Train subjects ids for split {split_counter}: {train_subjects}, train labels: {all_labels[train_index]}")
        print(f"Test subjects ids for split {split_counter}: {test_subjects}, test labels: {all_labels[test_index]}")

    # train_seizure_subjects, test_seizure_subjects = train_test_split(seizure_subjects, test_size=1-config['split_ratio'], random_state=split_seed)
    # train_bckg_subjects, test_bckg_subjects = train_test_split(bckg_subjects, test_size=1-config['split_ratio'], random_state=split_seed)
    # train_subjects = train_seizure_subjects + train_bckg_subjects
    # test_subjects = test_seizure_subjects + test_bckg_subjects

    # train events are all events from train subjects
        train_events = all_events[all_events['subject'].isin(train_subjects)]
    # test events are all events from test subjects
        test_events = all_events[all_events['subject'].isin(test_subjects)]

        if config.get('debug', False):
            print("\nReading and labeling events for split {}...".format(split_counter))
            if all_events.empty:
                print("No events found! Check your BIDS directory.")
                break
            print(f"Total events loaded for split {split_counter}: {len(all_events)}")
    
        # Create balanced epochs
        print(f"Creating balanced epochs for train set for split {split_counter}...")
        train_epochs = create_balanced_epochs(train_events, config)
        print(f"Creating balanced epochs for test set for split {split_counter}...")
        test_epochs = create_balanced_epochs(test_events, config)

        # train model
        model = train_model(train_epochs, config, split_counter, cv=True)
        # model = joblib.load('./models/cv_en_0728_093350_split_1.pkl')
        # test model
        test_model(model, test_epochs, config, split_counter, cv=True)
    # evaluate_recording(model, config)

    logging.shutdown()
    os.rename('temp.txt', log_path)

