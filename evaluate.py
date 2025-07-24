# Standard libraries
import os
import time
import math
import random
import joblib
import yaml
import sys
# Data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.axes import Axes
import colorsys
import logging

# Machine learning & classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.feature_based import Catch22Classifier

# Deep learning
import torch

# Domain-specific libraries
from mne.io import read_raw_edf
from mne import make_fixed_length_epochs, Annotations

# BIDS data handling
from bids import BIDSLayout

# Custom modules
from dataloader import (
    read_dataset,
    get_ids_from_filename,
    get_path_from_ids,
    extract_epochs,
    extract_event_info,
    get_data_from_epochs,
    read_ids_from_bids
)
from log import OutputCapture, StdoutCapture, StderrCapture
from detach_rocket.detach_classes import DetachRocket, DetachEnsemble
from analysis import Analyzer
from new_dataloader import read_all_events
from new_analysis import analyze_classification

# Scoring utilities
from timescoring.annotations import Annotation
from timescoring import scoring, visualization
from timescoring.scoring import SampleScoring, EventScoring

log_path = 'temp_eval.txt'
output_capture = OutputCapture(log_path, also_console=True)
sys.stdout = StdoutCapture(output_capture)
sys.stderr = StderrCapture(output_capture)


def evaluate_recording(edf_path, tsv_path, model_path, threshold, downsample=2.0, epoch_duration=10, epoch_overlap=0, plot=False, ss_path=None):

    model = joblib.load(model_path)
    raw_data = read_raw_edf(edf_path, preload=True)
    # desired_channels = raw_data.ch_names[:19]
    # raw_data.pick_channels(desired_channels, verbose=False)
    total_duration = raw_data._last_time
    fs = raw_data.info["sfreq"] / downsample # final sampling freq after downsample


    df_tsv = pd.read_csv(tsv_path, sep='\t')
    ref_events = []
    for _, row in df_tsv.iterrows():
        if row['eventType'] == 'bckg':
            continue
        start_time = int(row['onset'])
        end_time = int(row['onset'] + row['duration'])
        ref_events.append((start_time, end_time))

    n_samples = int(total_duration * fs)#?

    ref = Annotation(ref_events, fs, n_samples)

    events_info = extract_event_info(tsv_path, epoch_duration)
    epochs = extract_epochs(
        edf_path, events_info, downsample, 0, epoch_duration, epoch_overlap, inference=True) # inference mode: use all data
    segments = get_data_from_epochs(epochs)


    try:
        X_test = np.concatenate([s["epoch"] for s in segments]).astype(np.float32)
        y_test = np.concatenate([s["label"] for s in segments]).astype(int)
        time_start_test = np.concatenate([s["time_start"] for s in segments])
        time_end_test = np.concatenate([s["time_end"] for s in segments])
        #X_test = X_test[:, np.newaxis, :]
        del segments
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # y_pred = model.predict(X_test)
        # predictions = model.predict_proba(X_test)
        # yp = (predictions[:, 1] > threshold).astype(int) # threshold 
        # y_pred = model.label_encoder.inverse_transform(yp)
        y_pred = model.predict(X_test)
        
        hyp_events = []
        
        for pred, time_start, time_end in zip(y_pred, time_start_test, time_end_test):
            if pred == 1:
                hyp_events.append((time_start, time_end))

        hyp = Annotation(hyp_events, fs, n_samples)
        fig, ax = plt.subplots(2,1, figsize=(8, 4.8))
        # Compute sample-based scoring
        sample_scores = SampleScoring(ref, hyp)
        figSamples = visualization.plotSampleScoring(ref, hyp, ax=ax[0])
        
            
        # Compute event-based scoring
        param = scoring.EventScoring.Parameters(
        toleranceStart=30,
        toleranceEnd=60,
        minOverlap=0,
        maxEventDuration=100000,
        minDurationBetweenEvents=90)
        event_scores = scoring.EventScoring(ref, hyp, param)
        
        figEvents = visualization.plotEventScoring(ref, hyp, param, ax=ax[1])
        if plot:
            ax[0].figure.savefig(ss_path, bbox_inches='tight')
            
            
        # close the figures
        plt.close(figSamples)
        plt.close(figEvents)
    
        return sample_scores, event_scores
    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
def append_notnan_and_count_nan(value, lst, counter):
    if math.isnan(value):
        counter += 1
    else:
        lst.append(value)
    return counter



if __name__ == "__main__":
    config_id = 'test_0724_001701'
    config_file = './models/{}.yaml'.format(config_id)
    with open(config_file, 'r') as f: #read config file
        config = yaml.safe_load(f)
    dataset = "TUSZ" # "Siena" or "TUSZ"
    model_name = '{}.pkl'.format(config_id) 
    
    log_path = os.path.join(config['result_dir'], f'{config_id}_eval_{dataset}.txt')
    
    
    bids_root = config['bids_root'] # Replace with your actual path
    threshold = config['threshold']
    train_size = config['split_ratio']
    split_seed = config['split_seed']
    epoch_duration = config['epoch_duration'] # in seconds
    data_size = config['data_size']
    
    subject_ids = []
    for root, dirs, files in os.walk(bids_root):
        files.sort()
        for file in files:
            if file.endswith('.edf'):
                subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                subject_ids.append(subject_id)
    subject_ids = np.unique(subject_ids)
    
    if dataset == "TUSZ":
        all_events, (seizure_subjects, bckg_subjects) = read_all_events(config)
        
        train_seizure_subjects, test_seizure_subjects = train_test_split(seizure_subjects, test_size=1-config['split_ratio'], random_state=split_seed)
        train_bckg_subjects, test_bckg_subjects = train_test_split(bckg_subjects, test_size=1-config['split_ratio'], random_state=split_seed)
        train_subjects = train_seizure_subjects + train_bckg_subjects
        test_subjects = test_seizure_subjects + test_bckg_subjects
        train_subject_idx = train_subjects
        test_subject_idx = test_subjects
        print(f"Train subjects: {len(train_subjects)}")
        print(f"Test subjects: {len(test_subjects)}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda available: ", torch.cuda.is_available())
        print("device: ", device)
        
        model_path = os.path.join(config['model_dir'], model_name)
        model = joblib.load(model_path)
    
    if dataset == "Siena":
        data_size = 1
        test_segments, test_epoch_numbers_df = read_dataset(bids_root, epoch_duration, max_workers=2) # set max_workers to 1 for debugging
        test_subject_idx = subject_ids
        
        X_test = np.concatenate([s['epoch'] for s in test_segments]).astype(np.float32)
        y_test = np.concatenate([s['label'] for s in test_segments]).astype(int)
        
        del test_segments
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda available: ", torch.cuda.is_available())
        print("device: ", device)
        
        model_path = os.path.join(config['model_dir'], model_name)
        if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = joblib.load(f)
                print(f"Model loaded from: {model_path}")
        
        start_model_time = time.time()
        
        # model prediction on test set
        # predictions = model.predict_proba(X_test)
        # yp = (predictions[:, 1] > threshold).astype(int) # threshold = 0.5
        
        # y_pred = model.label_encoder.inverse_transform(yp)
        y_pred = model.predict(X_test)
        
        f1, precision, recall, accuracy, conf_mat, conf_mat_norm = analyze_classification(y_pred, y_test)
        
        end_model_time = time.time()
    
    if dataset == "test":
        data_size = 1
        seizure_epochs, non_seizure_epochs, bckg_epochs = [], [], [],
        recording_ids = read_ids_from_bids(bids_root)
        for ids in recording_ids:

            tsv_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="tsv")
            edf_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="edf")
            events_info = extract_event_info(tsv_path, 10)
            epochs = extract_epochs(edf_path, events_info, inference=True)
    
        
        test_segments = get_data_from_epochs(epochs)

        
        X_test = np.concatenate([s['epoch'] for s in test_segments]).astype(np.float32)
        y_test = np.concatenate([s['label'] for s in test_segments]).astype(int)
        
        device = torch.device("cuda")
        # device = torch.device("cpu")
        model_path = os.path.join(config['model_dir'], model_name)
        model = joblib.load(model_path)
        
        start_model_time = time.time()
        y_pred = model.predict(X_test)
        end_model_time = time.time()
        print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")
        
        
    ################## Evaluate the model #################
    
    start_model_time = time.time()
    
    sample_sensitivity, sample_precision, sample_f1, sample_fpRate, event_sensitivity, event_precision, event_f1, event_fpRate = [], [], [], [], [], [], [], []
    sample_precision_nan, sample_f1_nan, event_precision_nan, event_f1_nan = 0, 0, 0, 0
    
    subject_id_list = []
    bckg_counter,seiz_counter = 0, 0
    result_path = os.path.join(config['result_dir'], f"{model_name}_{dataset}_results.csv")
    result_dir = config['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    test_ids = []
    for root, dirs, files in os.walk(bids_root):
        files.sort()
        for file in files:
            if file.endswith('.edf'):
                subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                if subject_id in test_subject_idx:
                    test_ids.append({'subject_id': subject_id, 
                                     'session_id': session_id, 
                                     'task_id': task_id, 
                                     'run_id': run_id})
                    
    idx_cnt = 0
    for ids in test_ids:
        edf_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format = 'edf')
        tsv_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format = 'tsv')
        test_events_df = pd.read_csv(tsv_path, sep='\t')
                
        if "bckg" in test_events_df["eventType"].values:
            bckg_counter += 1
        if test_events_df["eventType"].str.contains("sz").any():
            seiz_counter += 1
                
        subject_id = ids['subject_id']
        session_id = ids['session_id']
        task_id = ids['task_id']
        run_id = ids['run_id']
        
    
        ss_path = os.path.join(
            config['result_dir'],
            f"{config_id}_{dataset}",
            f"{dataset}_sub-{subject_id}_ses-{session_id}_run-{run_id}.png"
        )
        
        img_dir = os.path.dirname(ss_path)
        os.makedirs(img_dir, exist_ok=True)
        
        sample_scores, event_scores = evaluate_recording(edf_path, tsv_path, model_path, threshold, plot=True, ss_path=ss_path)
        
        if sample_scores is None or event_scores is None:
            continue
        subject_id_list.append(subject_id)
        sample_sensitivity.append(sample_scores.sensitivity)
        event_sensitivity.append(event_scores.sensitivity)
        sample_precision.append(sample_scores.precision)
        event_precision.append(event_scores.precision)
        sample_f1.append(sample_scores.f1)
        event_f1.append(event_scores.f1)
        sample_fpRate.append(sample_scores.fpRate)
        event_fpRate.append(event_scores.fpRate)

        # save the sensitivity, precision, and f1-score of the samples and events as csv
        if idx_cnt % 10 == 0:
            results = pd.DataFrame({
                "subject_id": subject_id_list,
                'sample_sensitivity': sample_sensitivity,
                'sample_precision': sample_precision,
                'sample_f1': sample_f1,
                'sample_fpRate': sample_fpRate,
                'event_sensitivity': event_sensitivity,
                'event_precision': event_precision,
                'event_f1': event_f1,
                'event_fpRate': event_fpRate,
            })
            results.to_csv(result_path, index=False)
        idx_cnt += 1
    
    results = pd.DataFrame({
        "subject_id": subject_id_list,
        'sample_sensitivity': sample_sensitivity,
        'sample_precision': sample_precision,
        'sample_f1': sample_f1,
        'sample_fpRate': sample_fpRate,
        'event_sensitivity': event_sensitivity,
        'event_precision': event_precision,
        'event_f1': event_f1,
        'event_fpRate': event_fpRate,
    })
    results.to_csv(result_path, index=False)
    
        
    end_model_time = time.time()
    print(f"Model evaluation took: {end_model_time - start_model_time:.2f} seconds")
    print(f"Number of bckg recordings in test set: {bckg_counter}")
    print(f"Number of sz recordings in test set: {seiz_counter}")
    logging.shutdown()
    os.rename('temp_eval.txt', log_path)
        
        
        
    
    
    
    

