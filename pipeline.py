from bids import BIDSLayout
# import mne
from mne.io import read_raw_edf
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataloader import read_siena_dataset
from wavelet_utils import wavelet_decompose_channels_from_segment
from custom_utils import get_feature_matrix, get_labels_from_info
import torch
from sklearn import svm
from sktime.classification.kernel_based import RocketClassifier
#pip install numpy scikit-learn pyts torch matplotlib sktime==0.30.0
#pip install git+https://github.com/gon-uri/detach_rocket
from detach_rocket.detach_classes import DetachRocket
#from tsai.all import *
def plot_eeg_segment(segment_data, times, channel_name, event_info):
    """
    Plots a segment of EEG data.

    Parameters:
    - segment_data (ndarray): Time-series data for the segment.
    - times (ndarray): Time values corresponding to the segment.
    - channel_name (str): Name of the channel being plotted.
    """
    #highlight the seizure events
    # for event in seizure_events:
     

    plt.figure(figsize=(10, 4))
    plt.plot(times, segment_data.T)
    for _, event in event_info.iterrows():
        onset_event = event['onset']
        duration_event = event['duration']
        end_time_event = onset_event + duration_event
        if end_time_event > times[-1] or onset_event < times[0]:
            continue
        plt.axvspan(onset_event, end_time_event, color='green', alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'EEG Segment for Channel {channel_name}')
    plt.grid(True)
    # plt.show()

def roi_overlap_ratio(start_time_roi, end_time_roi, event_info):
    """
    Calculate the overlap ratio of an extracted ROI with reference to ground truth events.
    
    Parameters:
    - start_time_roi (float): The start time of the ROI (in seconds).
    - end_time_roi (float): The end time of the ROI (in seconds).
    - event_info (pandas.DataFrame): DataFrame containing 'onset', 'duration', and 'channel' of events.
    
    Returns:
    - overlap_ratios (list): List of overlap ratios for each event.
    """
    
    overlap_ratios = []
    
    for _, event in event_info.iterrows():
        onset_event = event['onset']
        duration_event = event['duration']
        end_time_event = onset_event + duration_event
        
        # Calculate overlap times
        overlap_start = max(start_time_roi, onset_event)
        overlap_end = min(end_time_roi, end_time_event)
        
        # If there is an overlap, calculate the overlap duration
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
        else:
            overlap_duration = 0
        
        # Calculate the overlap ratio
        event_duration = duration_event
        if event_duration > 0:
            overlap_ratio = overlap_duration / event_duration
        else:
            overlap_ratio = 0
        
        overlap_ratios.append(overlap_ratio)
    
    return overlap_ratios

if __name__ == "__main__":
    # dname = os.path.dirname(os.path.abspath(__file__))
    # bids_root = dname + '\BIDS_Siena' 
    bids_root = 'E:\BIDS_Siena' # Replace with your actual path
    #bids_root = 'E:\BIDS_CHB-MIT'

    # decom_wavelets = wavelet_decompose_channels_from_segment(segment, times, desired_channel, event_info, level=5, output=True)

    # plot_eeg_segment(segment, times, desired_channel, event_info)

    event_infos, segments = read_siena_dataset(bids_root, max_workers=2)

    # start_feature_time = time.time()
    # features = get_feature_matrix(segments)
    # end_feature_time = time.time()
    # print(f"Feature extraction took: {end_feature_time - start_feature_time:.2f} seconds")

    # train a SVM model to predict the labels based on the features
    start_model_time = time.time()

    # X = np.concatenate([f.ravel() for f in features]).reshape(-1, 1)
    # y = np.concatenate([f.ravel() for f in labels]).reshape(-1, 1)

    #only in desired channels?
    X = np.concatenate([s['epoch'] for s in segments]).astype(np.float32)
    y = np.concatenate([s['label'] for s in segments]).astype(np.float32)
    X = X[:, np.newaxis, :] # uncomment if use detach_rocket
    print(X.shape, y.shape)

    # In case of overflow
    del segments

    train_size = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
    del X, y
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available: ", torch.cuda.is_available())
    print("device: ", device)
    # Detach rocket model
    model = DetachRocket('pytorch_minirocket', num_kernels=10000, verbose=True, device=device) # multivariate; input_shape=(n_samples, n_channels, timestamps)

    # Rocket model from sktime
    #model = RocketClassifier(rocket_transform="minirocket", n_jobs=-1)

    model.fit(X_train, y_train)
    end_model_time = time.time()
    print(f"Model training took: {end_model_time - start_model_time:.2f} seconds")

    start_model_time = time.time()
    y_pred = model.predict(X_test)
    end_model_time = time.time()
    print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")

    accuracy = np.mean(y_pred == y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # plt.plot(times, predicted)
    # plt.show()



