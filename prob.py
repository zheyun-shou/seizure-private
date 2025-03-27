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


if __name__ == "__main__":
    dname = os.path.dirname(os.path.abspath(__file__))

    bids_root = 'F:\BIDS_TUSZ' # Replace with your actual path

    segments = read_siena_dataset(bids_root, max_workers=2) # set max_workers to 1 for debugging
    start_model_time = time.time()

    # In case of overflow
    threshold = 0.5
    train_size = 0.8
    
    # split the data into training and testing sets in subject-wise manner
    train_segments, test_segments = subject_wise_split(segments, train_ratio=train_size)
    del segments
    X_train = np.concatenate([s['epoch'] for s in train_segments]).astype(np.float32)
    y_train = np.concatenate([s['label'] for s in train_segments]).astype(int)
    X_test = np.concatenate([s['epoch'] for s in test_segments]).astype(np.float32)
    y_test = np.concatenate([s['label'] for s in test_segments]).astype(int)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load model
    model_name = 'en_d_mini_multi_tusz_221'
    model_path = 'D:/seizure/models/' + model_name + '.pkl'
    model = joblib.load(model_path)
    
    # start_model_time = time.time()
    
    predictions = model.predict(X_test)
    predictions = model.predict_proba(X_test)
    # save to csv
    result_path = f"D:/seizure/results/{model_name}_{threshold}_2/predictions.csv"
    pd.DataFrame(predictions).to_csv(result_path, index=False)
   
    # save label, time_start, time_end and ids of test_segments to csv
    test_seg_df = []
    for i, s in enumerate(test_segments):
        test_seg_df.append({'label': s['label'], 'time_start': s['time_start'], 'time_end': s['time_end'], 'subject_id': s['subject'], 'session_id': s['session_id'], 'task_id': s['task_id'], 'run_id': s['run_id']})
    test_seg_df_path = f"D:/seizure/results/{model_name}_{threshold}_2/test_segments.csv"
    pd.DataFrame(test_seg_df).to_csv(test_seg_df_path, index=False)

        
        

        
    
    # yp = (predictions[:, 1] > threshold).astype(int) # threshold = 0.8
    
    # y_pred = model.label_encoder.inverse_transform(yp)
    
    # end_model_time = time.time()
    # print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")
