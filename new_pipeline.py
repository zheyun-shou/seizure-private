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
from sktime.classification.feature_based import Catch22Classifier
import sys
import logging


# Import existing utilities
from new_dataloader import (
    load_epoch_data,
    load_evaluation_data_from_recording,
    read_all_events, 
    create_balanced_epochs,
    train_test_epochs_split, 
    find_matching_config,
    is_seizure_event
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
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
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
    
    print(f"Creating balanced epochs for train set...")
    train_epochs = create_balanced_epochs(train_events, config)
    print(f"Creating balanced epochs for test set...")
    test_epochs = create_balanced_epochs(test_events, config)


    print("\n=== Preprocess Pipeline Completed Successfully! ===")

    return train_epochs, test_epochs

def train_model(train_epochs, config):
    """Train the model."""

    # check if the config file exists
    if config.get('load_model', False):
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
        
    log_path = os.path.join(config['model_dir'], f'{config["model_name"]}_{config["timestamp"]}_log.txt')

    X_train, y_train = load_epoch_data(train_epochs, config, split_name='train')
    
    print(f"\n ===== Training Model ===== \n")

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    if config['model_type'] == 'detach_rocket':
        model = DetachRocket('pytorch_minirocket', num_kernels=config['num_kernels'], verbose=False, device=device) # multivariate; input_shape=(n_samples, n_channels, timestamps)
    elif config['model_type'] == 'detach_ensemble':
        model = DetachEnsemble(num_models=config['num_models'], num_kernels=config['num_kernels'], model_type='pytorch_minirocket', verbose=False)
    elif config['model_type'] == 'catch22':
        model = Catch22Classifier(random_state=42, n_jobs=-1)

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

    # save model config to the same directory
    # TODO: we need configuration, number of epochs, performance(terminal output).
    config_path = os.path.join(config['model_dir'], f"{model_name}.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_path}")

    print(f"\n ===== Training Model Completed ===== \n")

    return model

def test_model(model, test_epochs, config):
    """Evaluate the model."""
    X_test, y_test = load_epoch_data(test_epochs, config, split_name='test')
    print(f"\n ===== Evaluating Model on test set ===== \n")

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

        if is_seizure_event(recording_events, config):
            seiz_counter += 1
        else:
            bckg_counter += 1

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

class OutputCapture:
    def __init__(self, filename, also_console=True):
        self.filename = filename
        self.also_console = also_console
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Set up file logging
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filemode='w'
        )
        self.logger = logging.getLogger('output_capture')
    
    def write_stdout(self, text):
        if text.strip():  # Only log non-empty lines
            self.logger.info(f"STDOUT: {text.strip()}")
        if self.also_console:
            self.original_stdout.write(text)
    
    def write_stderr(self, text):
        if text.strip():
            self.logger.error(f"STDERR: {text.strip()}")
        if self.also_console:
            self.original_stderr.write(text)
    
    def flush(self):
        if self.also_console:
            self.original_stdout.flush()
            self.original_stderr.flush()

class StdoutCapture:
    def __init__(self, capture_obj):
        self.capture = capture_obj
    def write(self, text):
        self.capture.write_stdout(text)
    def flush(self):
        self.capture.flush()

class StderrCapture:
    def __init__(self, capture_obj):
        self.capture = capture_obj
    def write(self, text):
        self.capture.write_stderr(text)
    def flush(self):
        self.capture.flush()

# Set up output capture at the top of your main file
log_path = 'temp.txt'
output_capture = OutputCapture(log_path, also_console=True)
sys.stdout = StdoutCapture(output_capture)
sys.stderr = StderrCapture(output_capture)

if __name__ == "__main__":
    
    # logger = logging.getLogger(__name__)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

        config['timestamp'] = datetime.now().strftime("%m%d_%H%M%S") # add unique timestamp to the config
        # log_path = os.path.join(config['model_dir'], 'temp.txt')
        # logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO)

        
        config['dataset'] = config['bids_root'].split('/')[-1]

    os.makedirs(config['preprocessed_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)

    

    train_epochs, test_epochs = load_dataset(config)

    model = train_model(train_epochs, config)
    test_model(model, test_epochs, config)
    # evaluate_recording(model, config)
    # close the log file
    logging.shutdown()
    os.rename(log_path, os.path.join(config['model_dir'], f'{config["model_name"]}_{config["timestamp"]}_log.txt'))

