from bids import BIDSLayout
# import mne
from mne.io import read_raw_edf
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from dataloader import read_dataset, get_ids_from_filename, get_path_from_ids
from wavelet_utils import wavelet_decompose_channels_from_segment
from custom_utils import get_feature_matrix, get_labels_from_info, plot_confusion_matrix, print_confusion_matrix
import torch
from sklearn import svm
from sktime.classification.kernel_based import RocketClassifier
from detach_rocket.detach_classes import DetachRocket, DetachEnsemble
import joblib
from analysis import Analyzer
import random
from evaluate import evaluate_recording, append_notnan_and_count_nan
from pipeline import subject_wise_split
from dataloader import extract_event_info, extract_epochs, get_data_from_epochs

if __name__ == "__main__":
    dname = os.path.dirname(os.path.abspath(__file__))

    bids_root = 'F:\BIDS_TUSZ' # Replace with your actual path

    start_model_time = time.time()

    # In case of overflow
    threshold = 0.5
    train_size = 0.8
    downsample = 2
    epoch_duration = 10
    epoch_overlap = 0
    model_name = 'en_d_mini_multi_tusz_0406'
    data_size = 0.5 # 0.5 for TUSZ, 1.0 for Siena
    dataset = 'TUSZ' # TUSZ or Siena
    
    subject_ids = []
    for root, dirs, files in os.walk(bids_root):
        for file in files:
            if file.endswith('.edf'):
                subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                subject_ids.append(subject_id)
    subject_ids = np.unique(subject_ids)
    # only keep odd subject ids, data_size=0.5, only in TUSZ
    if data_size == 0.5:
        subject_ids = [s for s in subject_ids if int(s) % 2 == 1]
    
    
    train_subject_idx, test_subject_idx = train_test_split(subject_ids, train_size=train_size, random_state=42)
    # train_segments, train_epoch_numbers_df = read_dataset(bids_root, train_subject_idx, data_size=data_size, max_workers=2) # set max_workers to 1 for debugging
    # test_segments, test_epoch_numbers_df = read_dataset(bids_root, test_subject_idx, data_size=data_size, max_workers=2) # set max_workers to 1 for debugging

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load model
    
    model_path = 'D:/seizure/models/' + model_name + '.pkl'
    model = joblib.load(model_path)
    
    # get ids from test_subject_idx
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

    
    
    # start_model_time = time.time()
    
    for ids in test_ids:
        #print(ids)
        edf_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="edf")
        tsv_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="tsv")
        
        raw_data = read_raw_edf(edf_path, preload=True)
        total_duration = raw_data._last_time
        fs = raw_data.info["sfreq"] / downsample
        
        event_info = extract_event_info(tsv_path, epoch_duration)
        
        if event_info is not None:
            epochs = extract_epochs(edf_path, event_info, downsample, 0, epoch_duration, epoch_overlap, inference=True)
            if len(epochs) < 1 or epochs is None:
                print(f"No epochs found for {ids['subject_id']} {ids['session_id']} {ids['task_id']} {ids['run_id']}")
                continue
            segments = get_data_from_epochs(epochs)
        
        X_test = np.concatenate([s["epoch"] for s in segments]).astype(np.float32)
        y_test = np.concatenate([s["label"] for s in segments]).astype(int)
        time_start_test = np.concatenate([s["time_start"] for s in segments])
        time_end_test = np.concatenate([s["time_end"] for s in segments])
        
        del segments

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        predictions = model.predict_proba(X_test)
        
        subject_id, session_id, task_id, run_id = ids['subject_id'], ids['session_id'], ids['task_id'], ids['run_id']
        pred_path = f"D:/seizure/results/probs_{dataset}_{model_name}/sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_pred.csv"
        pred_dir = os.path.dirname(pred_path)
        os.makedirs(pred_dir, exist_ok=True)
        # save predictions to csv, including time_start and time_end, and label
        pred_df = []
        for i, p in enumerate(predictions):
            pred_df.append({'time_start': time_start_test[i], 'time_end': time_end_test[i], 'label': y_test[i], 'prediction': p[1]})
   
        pd.DataFrame(pred_df).to_csv(pred_path, index=False)
    

        
        

        
    
    # yp = (predictions[:, 1] > threshold).astype(int) # threshold = 0.8
    
    # y_pred = model.label_encoder.inverse_transform(yp)
    
    # end_model_time = time.time()
    # print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")
