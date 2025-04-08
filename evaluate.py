# Standard libraries
import os
import time
import math
import random
import joblib

# Data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.axes import Axes
import colorsys

# Machine learning & classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from sktime.classification.kernel_based import RocketClassifier

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
    get_data_from_epochs
)
from wavelet_utils import wavelet_decompose_channels_from_segment
from custom_utils import (
    get_feature_matrix,
    get_labels_from_info,
    plot_confusion_matrix,
    print_confusion_matrix
)
from detach_rocket.detach_classes import DetachRocket, DetachEnsemble
from analysis import Analyzer

# Scoring utilities
from timescoring.annotations import Annotation
from timescoring import scoring, visualization
from timescoring.scoring import SampleScoring, EventScoring

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
        predictions = model.predict_proba(X_test)
        yp = (predictions[:, 1] > threshold).astype(int) # threshold 
        y_pred = model.label_encoder.inverse_transform(yp)
        
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
    
    dataset = "Siena" # "Siena" or "TUSZ"
    bids_root = f'F:\BIDS_{dataset}' # Replace with your actual path
    threshold = 0.5
    train_size = 0.8
    model_name = 'en_d_mini_multi_tusz_0406'
    
    
    subject_ids = []
    for root, dirs, files in os.walk(bids_root):
        for file in files:
            if file.endswith('.edf'):
                subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                subject_ids.append(subject_id)
    subject_ids = np.unique(subject_ids)
    
    if dataset == "TUSZ":
        data_size = 0.5
        # only keep odd subject ids, data_size=0.5, only in TUSZ
        if data_size == 0.5:
            subject_ids = [s for s in subject_ids if int(s) % 2 == 1]
            
        train_subject_idx, test_subject_idx = train_test_split(subject_ids, train_size=train_size, random_state=42)
        train_segments, train_epoch_numbers_df = read_dataset(bids_root, train_subject_idx, data_size=data_size, max_workers=2) # set max_workers to 1 for debugging
        test_segments, test_epoch_numbers_df = read_dataset(bids_root, test_subject_idx, data_size=data_size, max_workers=2) # set max_workers to 1 for debugging
        
        
        print(f"Train subjects: {train_subject_idx}")
        print(f"Test subjects: {test_subject_idx}")
        
        X_train = np.concatenate([s['epoch'] for s in train_segments]).astype(np.float32)
        y_train = np.concatenate([s['label'] for s in train_segments]).astype(int)
        X_test = np.concatenate([s['epoch'] for s in test_segments]).astype(np.float32)
        y_test = np.concatenate([s['label'] for s in test_segments]).astype(int)
        
        del train_segments, test_segments
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda available: ", torch.cuda.is_available())
        print("device: ", device)
        
        model_path = 'D:/seizure/models/' + model_name + '.pkl'
        model = joblib.load(model_path)
        
        # start_model_time = time.time()
        
        # model prediction on test set
        # predictions = model.predict_proba(X_test)
        # yp = (predictions[:, 1] > threshold).astype(int) # threshold = 0.5
        
        # y_pred = model.label_encoder.inverse_transform(yp)
        
        # analyzer = Analyzer(print_conf_mat=True)
        # analyzer.analyze_classification(y_pred, y_test, ['normal', 'seizure'])
        # accuracy = np.mean(y_pred == y_test)
        # print(f"Epoch-wise model accuracy on TUSZ, test set: {accuracy:.2f}")
        
        #prediction on train set, for overfitting check
        # predictions_train = model.predict_proba(X_train)
        # yp_train = (predictions_train[:, 1] > threshold).astype(int) # threshold = 0.5
        # y_pred_train = model.label_encoder.inverse_transform(yp_train)
        
        # analyzer_train = Analyzer(print_conf_mat=True)
        # analyzer_train.analyze_classification(y_pred_train, y_train, ['normal', 'seizure'])
        # accuracy_train = np.mean(y_pred_train == y_train)
        # print(f"Epoch-wise model accuracy on TUSZ, train set: {accuracy_train:.2f}")
        
        # end_model_time = time.time()
        # print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")
    
    if dataset == "Siena":
        data_size = 1
        test_segments, test_epoch_numbers_df = read_dataset(bids_root, data_size=data_size, max_workers=2) # set max_workers to 1 for debugging
        test_subject_idx = subject_ids
        
        X_test = np.concatenate([s['epoch'] for s in test_segments]).astype(np.float32)
        y_test = np.concatenate([s['label'] for s in test_segments]).astype(int)
        
        del test_segments
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda available: ", torch.cuda.is_available())
        print("device: ", device)
        
        model_path = 'D:/seizure/models/' + model_name + '.pkl'
        model = joblib.load(model_path)
        
        start_model_time = time.time()
        
        # model prediction on test set
        predictions = model.predict_proba(X_test)
        yp = (predictions[:, 1] > threshold).astype(int) # threshold = 0.5
        
        y_pred = model.label_encoder.inverse_transform(yp)
        
        analyzer = Analyzer(print_conf_mat=True)
        analyzer.analyze_classification(y_pred, y_test, ['normal', 'seizure'])
        accuracy = np.mean(y_pred == y_test)
        print(f"Epoch-wise model accuracy on Siena: {accuracy:.2f}")
            
    ################## Evaluate the model #################
    
    start_model_time = time.time()
    
    sample_sensitivity, sample_precision, sample_f1, event_sensitivity, event_precision, event_f1 = [], [], [], [], [], []
    sample_precision_nan, sample_f1_nan, event_precision_nan, event_f1_nan = 0, 0, 0, 0
    recording_counter = 0
    subject_id_list = []
    bckg_counter,seiz_counter = 0, 0
    result_path = f"D:/seizure/results/{model_name}_{dataset}/results.csv"
    result_dir = os.path.dirname(result_path)
    os.makedirs(result_dir, exist_ok=True)
    
    test_ids = []
    for root, dirs, files in os.walk(bids_root):
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
                
        for i, row in test_events_df.iterrows():
            if row["eventType"] == "bckg":
                bckg_counter += 1
            elif "sz" in row["eventType"]:
                seiz_counter += 1
                
        subject_id = ids['subject_id']
        session_id = ids['session_id']
        task_id = ids['task_id']
        run_id = ids['run_id']
    
        ss_path = f"D:/seizure/results/{model_name}_{dataset}/{dataset}_sub-{subject_id}_ses-{session_id}_{task_id}_run-{run_id}.png"
        
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

        # save the sensitivity, precision, and f1-score of the samples and events as csv
        if idx_cnt % 10 == 0:
            results = pd.DataFrame({
                "subject_id": subject_id_list,
                'sample_sensitivity': sample_sensitivity,
                'sample_precision': sample_precision,
                'sample_f1': sample_f1,
                'event_sensitivity': event_sensitivity,
                'event_precision': event_precision,
                'event_f1': event_f1,
            })
            results.to_csv(result_path, index=False)
        idx_cnt += 1
    
    results = pd.DataFrame({
        "subject_id": subject_id_list,
        'sample_sensitivity': sample_sensitivity,
        'sample_precision': sample_precision,
        'sample_f1': sample_f1,
        'event_sensitivity': event_sensitivity,
        'event_precision': event_precision,
        'event_f1': event_f1,
    })
    results.to_csv(result_path, index=False)
    
        
    end_model_time = time.time()
    print(f"Model evaluation took: {end_model_time - start_model_time:.2f} seconds")
    print(f"Number of bckg: {bckg_counter}")
    print(f"Number of seiz: {seiz_counter}")
            
        
        
        
    
    
    
    

