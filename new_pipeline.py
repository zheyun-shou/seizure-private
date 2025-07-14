#!/usr/bin/env python3
"""
New pipeline for seizure detection.
This replaces the old approach of loading data from segments.
"""

import os
import time
from datetime import datetime
import pandas as pd
import yaml
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import joblib
import random

# Import existing utilities
from new_dataloader import (
    load_epoch_data,
    load_evaluation_data_from_recording,
    read_all_events, 
    create_balanced_epochs,
    train_test_epochs_split, 
    find_matching_config,
)
from new_analysis import analyze_classification, timescoring_analysis
from mne_utils import setup_mne_for_processing
from detach_rocket.detach_classes import DetachRocket, DetachEnsemble

def load_dataset(config):
    """Load the dataset."""

    # Configure MNE verbosity
    setup_mne_for_processing(verbose=config.get('mne_verbose', False))

    print("\n=== Preprocess Pipeline with Balanced Epochs ===")
    
    # Step 1: Read and label all events
    all_events, (seizure_subjects, bckg_subjects) = read_all_events(config)
    
    split_seed = config.get('split_seed')
    train_seizure_subjects, test_seizure_subjects = train_test_split(seizure_subjects, test_size=1-config['split_ratio'], random_state=split_seed)
    train_bckg_subjects, test_bckg_subjects = train_test_split(bckg_subjects, test_size=1-config['split_ratio'], random_state=split_seed)
    train_subjects = train_seizure_subjects + train_bckg_subjects
    test_subjects = test_seizure_subjects + test_bckg_subjects
    # train events are all events from train subjects
    train_events = all_events[all_events['subject'].isin(train_subjects)]
    # test events are all events from test subjects
    test_events = all_events[all_events['subject'].isin(test_subjects)]

    
    if config.get('debug', False):
        print("\nReading and labeling events...")
        if all_events.empty:
            print("No events found! Check your BIDS directory.")
            return
        print(f"Total events loaded: {len(all_events)}")
    
    # Step 3: Create balanced epochs
    print("\nCreating balanced epochs...")
    train_epochs = create_balanced_epochs(train_events, config)
    test_epochs = create_balanced_epochs(test_events, config)

    # Step 4: Split epochs into train and test
    print("\nSplitting epochs into train and test...")
    train_epochs, test_epochs = train_test_epochs_split(epochs, config)

    print("\n=== Preprocess Pipeline Completed Successfully! ===")

    return train_epochs, test_epochs

def train_model(train_epochs, config):
    """Train the model."""

    # check if the config file exists
    if config.get('load_model', False):
        config_file = find_matching_config(config, config['model_dir'])
        if config_file is not None:
            print(f"Loading model...")
            timestamp = config_file['timestamp']
            data_file = os.path.join(config['model_dir'], f'{config["model_name"]}_{timestamp}.pkl')
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    model = joblib.load(f)
                print(f"Model loaded from: {data_file}")
                return model
            else:
                print(f"Model file not found: {data_file}")

    X_train, y_train = load_epoch_data(train_epochs, config, split_name='train')
    
    print(f"\n ===== Training Model ===== \n")

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Detach rocket model
    model = DetachRocket('pytorch_minirocket', num_kernels=config['num_kernels'], verbose=False, device=device) # multivariate; input_shape=(n_samples, n_channels, timestamps)

    start_training_time = time.time()
    model.fit(X_train, y_train)
    end_training_time = time.time()
    print(f"Model training completed in: {end_training_time - start_training_time:.2f} seconds")
    
    # save model
    model_name = "{}_{}".format(config['model_name'], config['timestamp'])
    model_path = os.path.join(config['model_dir'], f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        joblib.dump(model, f)
    print(f"Model saved to: {model_path}")

    # save config to the same directory
    config_path = os.path.join(config['model_dir'], f"{model_name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_path}")

    print(f"\n ===== Training Model Completed ===== \n")

    return model

def test_model(model, test_epochs, config):
    """Evaluate the model."""
    X_test, y_test = load_epoch_data(test_epochs, config, split_name='test')
    print(f"\n ===== Evaluating Model ===== \n")

    # evaluate the model
    y_pred = model.predict(X_test)
    print(f"Test set size: {len(X_test)}")

    f1, precision, recall, accuracy, conf_mat, conf_mat_norm = analyze_classification(y_pred, y_test)

    print(f"\n ===== Evaluating Model Completed ===== \n")

    return y_pred

def evaluate_recording(model, config):
    """Evaluate the model on a single recording."""
    recording_wise_data = load_evaluation_data_from_recording(config)
    summary_list = []
    bckg_counter = 0
    seiz_counter = 0

    for recording_key, data in recording_wise_data.items():
        # Extract data from the dictionary
        epoch_data = data['epoch_data']
        labels = data['labels']
        intervals = data['interval']
        recording_events = data['events']
        recording_ids = data['recording_ids']
        fs = data['fs']
        
        # do model prediction
        y_pred = model.predict(epoch_data)

        if "bckg" in recording_events["eventType"].values:
            bckg_counter += 1
        
        # Check if any seizure event types are present
        has_seizure = False
        for seizure_type in config['seizure_event_type']:
            if recording_events["eventType"].str.contains(seizure_type, case=False).any():
                has_seizure = True
                break
        if has_seizure:
            seiz_counter += 1

        recording_ref_events = [(int(row['onset']), int(row['onset'] + row['duration'])) \
                                for _, row in recording_events.iterrows() if row['eventType'] != 'bckg']
        recording_hyp_events = [interval for interval, pred in zip(intervals, y_pred) if pred == 1]
        n_samples = int(recording_events['recordingDuration'][0] * fs)
        summary = timescoring_analysis(recording_ref_events, recording_hyp_events, fs, n_samples, recording_ids, config)
        summary_list.append(summary)

    summary_df = pd.DataFrame(summary_list)
    summary_path = os.path.join(config['result_dir'], f'{config["model_name"]}_{config["dataset"]}', 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    print(f"Number of bckg recordings in test set: {bckg_counter}")
    print(f"Number of sz recordings in test set: {seiz_counter}")


if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S") # add unique timestamp to the config
        config['dataset'] = config['bids_root'].split('/')[-1]
    os.makedirs(config['preprocessed_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)

    train_epochs, test_epochs = load_dataset(config)

    model = train_model(train_epochs, config)
    test_model(model, test_epochs, config)
    # evaluate_recording(model, config)
