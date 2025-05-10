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
#pip install numpy scikit-learn pyts torch matplotlib sktime==0.30.0
#pip install git+https://github.com/gon-uri/detach_rocket
from detach_rocket.detach_classes import DetachRocket, DetachEnsemble
import joblib
from analysis import Analyzer
import random
from evaluate import evaluate_recording, append_notnan_and_count_nan
from scipy.interpolate import griddata


if __name__ == "__main__":
    
    #load model
    
    model_name = '0426_en_mini_datasize0.2'
    model_path = 'D:/seizure/models/' + model_name + '.pkl'
    model = joblib.load(model_path)
    
    channel_relevance_matrix = model.estimate_channel_relevance()
    print(channel_relevance_matrix)
    print("Channel relevance matrix shape:", channel_relevance_matrix.shape)
    raw_data = read_raw_edf("F:\BIDS_TUSZ\sub-000\ses-00\eeg\sub-000_ses-00_task-szMonitoring_run-01_eeg.edf", preload=False, verbose=False)
    
    channel_list = []
    for ch in raw_data.ch_names:
        # map the channel name to the relevance matrix
        channel_list.append(ch)
        ch_idx = raw_data.ch_names.index(ch)
        ch_relevance = channel_relevance_matrix[ch_idx]
        ch_without_avg = ch.replace("-Avg", "")
        print(f"\"{ch_without_avg}\": {ch_relevance},")
    
    

