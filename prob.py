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
from dataloader import read_siena_dataset, get_ids_from_filename, get_path_from_ids
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

    segments = read_siena_dataset(bids_root, max_workers=2) # set max_workers to 1 for debugging
    start_model_time = time.time()

    # In case of overflow
    threshold = 0.5
    train_size = 0.8
    downsample = 2
    epoch_duration = 10
    epoch_overlap = 0
    
    # split the data into training and testing sets in subject-wise manner
    train_segments, test_segments, train_ids, test_ids = subject_wise_split(segments, train_ratio=train_size, return_ids=True)
    del segments
    # X_test = np.concatenate([s['epoch'] for s in test_segments]).astype(np.float32)
    # y_test = np.concatenate([s['label'] for s in test_segments]).astype(int)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load model
    model_name = 'en_d_mini_multi_tusz_221'
    model_path = 'D:/seizure/models/' + model_name + '.pkl'
    model = joblib.load(model_path)
    
    # start_model_time = time.time()
    
    for ids in test_ids:
        print(ids)
        edf_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="edf")
        tsv_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="tsv")
        
        raw_data = read_raw_edf(edf_path, preload=True)
        total_duration = raw_data._last_time
        fs = raw_data.info["sfreq"] / downsample
        
        seizure_info = extract_event_info(tsv_path, epoch_duration, filter=["sz", "seiz"])
        bckg_info = extract_event_info(tsv_path, epoch_duration, filter=["bckg"])
        
        if seizure_info is not None:
            seizure_epochs = extract_epochs(
                edf_path, seizure_info, downsample, 0, epoch_duration, epoch_overlap, info_type="seizure", inference=True)
            segments = get_data_from_epochs(seizure_epochs)
        else:
            bckg_epochs = extract_epochs(
                edf_path, bckg_info, downsample, 0, epoch_duration, epoch_overlap, info_type="non-seizure", inference=True)
            segments = get_data_from_epochs(bckg_epochs)
        
        X_test = np.concatenate([s["epoch"] for s in segments]).astype(np.float32)
        y_test = np.concatenate([s["label"] for s in segments]).astype(int)
        time_start_test = np.concatenate([s["time_start"] for s in segments])
        time_end_test = np.concatenate([s["time_end"] for s in segments])
        
        del segments

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        predictions = model.predict_proba(X_test)
        
        subject_id, session_id, task_id, run_id = ids['subject_id'], ids['session_id'], ids['task_id'], ids['run_id']
        pred_path = f"D:/seizure/results/recording_predictions/sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_pred.csv"
    
        # save predictions to csv, including time_start and time_end, and label
        pred_df = []
        for i, p in enumerate(predictions):
            pred_df.append({'time_start': time_start_test[i], 'time_end': time_end_test[i], 'label': y_test[i], 'prediction': p[1]})
   
        pd.DataFrame(pred_df).to_csv(pred_path, index=False)
    

        
        

        
    
    # yp = (predictions[:, 1] > threshold).astype(int) # threshold = 0.8
    
    # y_pred = model.label_encoder.inverse_transform(yp)
    
    # end_model_time = time.time()
    # print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")
