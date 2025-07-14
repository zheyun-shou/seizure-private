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
from sktime.classification.feature_based import Catch22Classifier


def subject_wise_split(segments, train_ratio=0.8, return_ids=False):
    # use np.unique to get the unique subjects
    subjects = np.concatenate([s['subject'] for s in segments]) #array
    unique_subject = np.unique(subjects)
    
    train_subject_idx, test_subject_idx = train_test_split(unique_subject, train_size=train_ratio, random_state=42) 
    train_segments = []
    test_segments = []
    for s in segments:
        if s['subject'][0] in train_subject_idx:
            train_segments.append(s)
        else:
            test_segments.append(s)
    if return_ids:
        train_ids = [{"subject_id": s["subject"][0],
                      "session_id": s["session_id"][0],
                      "task_id": s["task_id"][0],
                      "run_id": s["run_id"][0]} for s in train_segments]
        test_ids = [{"subject_id": s["subject"][0],
                      "session_id": s["session_id"][0],
                      "task_id": s["task_id"][0],
                      "run_id": s["run_id"][0]} for s in test_segments]
        return train_segments, test_segments, train_ids, test_ids
    else:
        return train_segments, test_segments


if __name__ == "__main__":
    # dname = os.path.dirname(os.path.abspath(__file__))
    # bids_root = dname + '\BIDS_Siena' 
    
    bids_root = 'F:\BIDS_TUSZ' # Replace with your actual path
    threshold = 0.5
    train_size = 0.8
    data_size = 0.5
    epoch_duration = 10
    rnd_seed = 40 
    model_name = '0529_de_mini_odd'


    # decom_wavelets = wavelet_decompose_channels_from_segment(segment, times, desired_channel, event_info, level=5, output=True)

    subject_ids = []
    for root, dirs, files in os.walk(bids_root):
        files.sort()
        for file in files:
            if file.endswith('.edf'):
                subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                subject_ids.append(subject_id)
    subject_ids = np.unique(subject_ids)
    # only keep odd subject ids, data_size=0.5, only in TUSZ
    if data_size == 0.5:
        subject_ids = [s for s in subject_ids if int(s) % 2 == 1]
    
    # shuffle subject ids and take the first 0.6 of them
    # random.seed(rnd_seed)
    # random.shuffle(subject_ids)
    # subject_ids = subject_ids[:int(len(subject_ids) * data_size)] 
        
    train_subject_idx, test_subject_idx = train_test_split(subject_ids, train_size=train_size, random_state=42)
    train_segments, train_epoch_numbers_df = read_dataset(bids_root, epoch_duration, train_subject_idx, max_workers=2) # set max_workers to 1 for debugging
    
    print("train subjects:")
    print(train_subject_idx)
    print("test subjects:")
    print(test_subject_idx)
    print("train epochs:")
    print(train_epoch_numbers_df)


    # start_feature_time = time.time()
    # features = get_feature_matrix(segments)
    # end_feature_time = time.time()
    # print(f"Feature extraction took: {end_feature_time - start_feature_time:.2f} seconds")


    X_train = np.concatenate([s['epoch'] for s in train_segments]).astype(np.float32)
    y_train = np.concatenate([s['label'] for s in train_segments]).astype(int)

    del train_segments
  
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available: ", torch.cuda.is_available())
    print("device: ", device)
    
    start_model_time = time.time()
    # catch22 model
    # model = Catch22Classifier(random_state=42, n_jobs=-1)
    # Detach rocket model
    model = DetachRocket('pytorch_minirocket', num_kernels=10000, verbose=False, device=device) # multivariate; input_shape=(n_samples, n_channels, timestamps)

    # Rocket model from sktime
    #model = RocketClassifier(rocket_transform="minirocket", n_jobs=-1)

    # Detach minirocket ensemble model
    # model = DetachEnsemble(num_models=10, num_kernels=10000, model_type='pytorch_minirocket', verbose=False)
    
    model.fit(X_train, y_train)
    
    end_model_time = time.time()
    print(f"Model training took: {end_model_time - start_model_time:.2f} seconds")
    
    # save model
    
    joblib.dump(model, 'D:/seizure/models/' + model_name + '.pkl')

    # load model
    # model_name = '0511_catch22_datasize0.5_2'
    # model_path = 'D:/seizure/models/' + model_name + '.pkl'
    # model = joblib.load(model_path)
    
    start_model_time = time.time()
    
    test_segments, test_epoch_numbers_df = read_dataset(bids_root, epoch_duration, test_subject_idx, max_workers=2) # set max_workers to 1 for debugging
    X_test = np.concatenate([s['epoch'] for s in test_segments]).astype(np.float32)
    y_test = np.concatenate([s['label'] for s in test_segments]).astype(int)
    del test_segments
    
    print("test epochs:")
    print(test_epoch_numbers_df)
    # model prediction on test set
    # predictions = model.predict_proba(X_test)
    # yp = (predictions[:, 1] > threshold).astype(int) # threshold = 0.5
    
    # y_pred = model.label_encoder.inverse_transform(yp)
    y_pred = model.predict(X_test)
    
    analyzer = Analyzer(print_conf_mat=True)
    analyzer.analyze_classification(y_pred, y_test, ['normal', 'seizure'])
    accuracy = np.mean(y_pred == y_test)
    print(f"Epoch-wise model accuracy on test set: {accuracy:.2f}")
    
    del X_test, y_test

    #model prediction on train set, for overfitting check
    # predictions_train = model.predict_proba(X_train)
    # yp_train = (predictions_train[:, 1] > threshold).astype(int) # threshold = 0.5
    # y_pred_train = model.label_encoder.inverse_transform(yp_train)
    
    # y_pred_train = model.predict(X_train)
    
    # analyzer = Analyzer(print_conf_mat=True)
    # analyzer.analyze_classification(y_pred_train, y_train, ['normal', 'seizure'])
    # accuracy_train = np.mean(y_pred_train == y_train)
    # print(f"Epoch-wise model accuracy on train set: {accuracy_train:.2f}")
    
    
    end_model_time = time.time()
    print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")


    
    


