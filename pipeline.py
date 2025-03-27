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
#pip install numpy scikit-learn pyts torch matplotlib sktime==0.30.0
#pip install git+https://github.com/gon-uri/detach_rocket
from detach_rocket.detach_classes import DetachRocket, DetachEnsemble
import joblib
from analysis import Analyzer
import random
from evaluate import evaluate_recording, append_notnan_and_count_nan


def subject_wise_split(segments, train_ratio=0.8):
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
    return train_segments, test_segments

if __name__ == "__main__":
    dname = os.path.dirname(os.path.abspath(__file__))
    # bids_root = dname + '\BIDS_Siena' 
    # bids_root = dname + '\BIDS_TUSZ'
    bids_root = 'F:\BIDS_TUSZ' # Replace with your actual path
    #bids_root = 'E:\BIDS_CHB-MIT'

    # decom_wavelets = wavelet_decompose_channels_from_segment(segment, times, desired_channel, event_info, level=5, output=True)

    # plot_eeg_segment(segment, times, desired_channel, event_info)

    segments = read_siena_dataset(bids_root, max_workers=2) # set max_workers to 1 for debugging

    # start_feature_time = time.time()
    # features = get_feature_matrix(segments)
    # end_feature_time = time.time()
    # print(f"Feature extraction took: {end_feature_time - start_feature_time:.2f} seconds")

    # train a SVM model to predict the labels based on the features
    start_model_time = time.time()

    # X = np.concatenate([f.ravel() for f in features]).reshape(-1, 1)
    # y = np.concatenate([f.ravel() for f in labels]).reshape(-1, 1)

    #only in desired channels?
    # X = np.concatenate([s['epoch'] for s in segments]).astype(np.float32)
    # y = np.concatenate([s['label'] for s in segments]).astype(int)
    #X = X[:, np.newaxis, :] # uncomment if use detach_rocket
    #print(X.shape, y.shape)

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
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
    # del X, y
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda available: ", torch.cuda.is_available())
    print("device: ", device)
    # Detach rocket model
    #model = DetachRocket('pytorch_minirocket', num_kernels=10000, verbose=True, device=device) # multivariate; input_shape=(n_samples, n_channels, timestamps)

    # Rocket model from sktime
    #model = RocketClassifier(rocket_transform="minirocket", n_jobs=-1)

    # Detach ensemble model
    # model = DetachEnsemble(num_models=10, num_kernels=10000, model_type='pytorch_minirocket', verbose=True)
    # model.fit(X_train, y_train)
    
    end_model_time = time.time()
    print(f"Model training took: {end_model_time - start_model_time:.2f} seconds")
    
    # save model
    # model_name = 'en_d_mini_multi_tusz_221'
    # joblib.dump(model, 'D:/seizure/models/' + model_name + '.pkl')

    # load model
    model_name = 'en_d_mini_multi_tusz_221'
    model_path = 'D:/seizure/models/' + model_name + '.pkl'
    model = joblib.load(model_path)
    
    start_model_time = time.time()
    
    predictions = model.predict(X_test)
    predictions = model.predict_proba(X_test)
    #print(predictions)
    yp = (predictions[:, 1] > threshold).astype(int) # threshold = 0.8
    
    y_pred = model.label_encoder.inverse_transform(yp)
    
    end_model_time = time.time()
    print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")

    analyzer = Analyzer(print_conf_mat=True)
    analyzer.analyze_classification(y_pred, y_test, ['normal', 'seizure'])
    accuracy = np.mean(y_pred == y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    
    ################## Evaluate the model on the test segments, don't forget to change load model #################
    
    start_model_time = time.time()
    
    sample_sensitivity, sample_precision, sample_f1, event_sensitivity, event_precision, event_f1 = [], [], [], [], [], []
    sample_precision_nan, sample_f1_nan, event_precision_nan, event_f1_nan = 0, 0, 0, 0
    recording_counter = 0
    subject_id_list = []
    bckg_counter,seiz_counter = 0, 0
    # read subject of test segments
    test_subject_list = []
    for test_s in test_segments:
        test_subject_list.append(test_s['subject'][0])
    test_subject_list = np.unique(test_subject_list)
        # read all the edf and tsv files under the test subject directory
    for test_subject in test_subject_list:
        subject_root = os.path.join(bids_root, f'sub-{test_subject}')
        for root, dirs, files in os.walk(subject_root):
            for file in files:
                if not file.endswith('.edf'):
                    continue # reduce repetition, we only need filename
                subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                ids = {
                        'subject_id': subject_id,
                        'session_id': session_id,
                        'task_id': task_id,
                        'run_id': run_id,
                    }
                edf_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format = 'edf')
                tsv_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format = 'tsv')
                test_events_df = pd.read_csv(tsv_path, sep='\t')
                
                for i, row in test_events_df.iterrows():
                    if row["eventType"] == "bckg":
                        bckg_counter += 1
                    elif row["eventType"] == "sz":
                        seiz_counter += 1
                    
            
                ss_path = f"D:/seizure/results/{model_name}_{threshold}_2/TUSZ_sub-{subject_id}_ses-{session_id}_{task_id}_run-{run_id}_sample_scoring.png"
                
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
    result_path = f"D:/seizure/results/{model_name}_{threshold}_2/results.csv"
    result_dir = os.path.dirname(result_path)
    os.makedirs(result_dir, exist_ok=True)
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

    
    




    # plt.plot(times, predicted)
    # plt.show()



