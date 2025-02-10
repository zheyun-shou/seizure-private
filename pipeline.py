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
from dataloader import read_siena_dataset
from wavelet_utils import wavelet_decompose_channels_from_segment
from custom_utils import get_feature_matrix, get_labels_from_info, plot_confusion_matrix, print_confusion_matrix
import torch
from sklearn import svm
from sktime.classification.kernel_based import RocketClassifier
#pip install numpy scikit-learn pyts torch matplotlib sktime==0.30.0
#pip install git+https://github.com/gon-uri/detach_rocket
from detach_rocket.detach_classes import DetachRocket
import joblib
from analysis import Analyzer
import random


def subject_wise_split(segments, train_ratio=0.8):
    # use np.unique to get the unique subjects
    subjects = np.concatenate([s['subject'] for s in segments]) #array
    #print(subjects)
    random.shuffle(subjects)
    train_subjects = subjects[:int(len(subjects) * train_ratio)] # array
    train_segments = []
    test_segments = []
    for segment in segments:
        if np.unique(segment['subject']) in train_subjects:
            train_segments.append(segment)
        else:
            test_segments.append(segment)
    return train_segments, test_segments

if __name__ == "__main__":
    dname = os.path.dirname(os.path.abspath(__file__))
    # bids_root = dname + '\BIDS_Siena' 
    # bids_root = dname + '\BIDS_TUSZ'
    bids_root = 'E:\BIDS_TUSZ' # Replace with your actual path
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
    y = np.concatenate([s['label'] for s in segments]).astype(int)
    #X = X[:, np.newaxis, :] # uncomment if use detach_rocket
    print(X.shape, y.shape)

    # In case of overflow
    

    train_size = 0.8
    # split the data into training and testing sets in subject-wise manner


    train_segments, test_segments = subject_wise_split(segments, train_ratio=0.8)
    del segments
    X_train = np.concatenate([s['epoch'] for s in train_segments]).astype(np.float32)
    y_train = np.concatenate([s['label'] for s in train_segments]).astype(int)
    X_test = np.concatenate([s['epoch'] for s in test_segments]).astype(np.float32)
    y_test = np.concatenate([s['label'] for s in test_segments]).astype(int)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
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

    analyzer = Analyzer(print_conf_mat=True)
    analyzer.analyze_classification(y_pred, y_test, ['normal', 'seizure'])
    accuracy = np.mean(y_pred == y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # save model
    joblib.dump(model, 'D:/seizure/models/d_mini_multivariate_tusz_sub.pkl')



    # plt.plot(times, predicted)
    # plt.show()



