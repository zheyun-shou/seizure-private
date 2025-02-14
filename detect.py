import os
import joblib
import numpy as np
import pandas as pd
import torch
import time
from mne.io import read_raw_edf
from mne import make_fixed_length_epochs
from mne import Annotations
from dataloader import extract_epochs, get_path_from_ids, extract_event_info




def detect_recording(edf_path, tsv_path, model_path, downsample=2.0, epoch_duration=10, epoch_overlap=5):
    
    start_model_time = time.time()

    # Load the saved model
    model = joblib.load(model_path)

    event_info = extract_event_info(tsv_path)

    # Create epochs (no event info provided, so treat the entire recording as one segment)
    #total_duration = raw_data._last_time
    
    epochs = extract_epochs(edf_path, event_info, downsample, inference=True)
    segments = []

    # Convert epochs to array for model
    for ep in epochs:
        epoch = ep.get_data() # shape (n_epochs, n_channels, n_times)
        epoch_labels = ep.metadata["label"].to_numpy()
        n_epochs, n_channels, n_times = epoch.shape
        epoch = epoch.reshape(n_epochs * n_channels, n_times)
        epoch_labels = np.tile(ep.metadata["label"], (n_channels))

        segments.append({"epoch": epoch, "label": epoch_labels})  

    X_test = np.concatenate([s["epoch"] for s in segments]).astype(np.float32)
    y_test = np.concatenate([s["label"] for s in segments]).astype(int)
    X_test = X_test[:, np.newaxis, :]

    del segments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred = model.predict(X_test)
    
    end_model_time = time.time()
    print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")

    return y_pred

if __name__ == "__main__":
    
    edf_path = "E:\BIDS_Siena\sub-10\ses-01\eeg\sub-10_ses-01_task-szMonitoring_run-00_eeg.edf"
    tsv_path = "E:\BIDS_Siena\sub-10\ses-01\eeg\sub-10_ses-01_task-szMonitoring_run-00_events.tsv"
    model_path = "D:\seizure\models\detach_minirocket.pkl"
    predictions = detect_recording(edf_path, tsv_path, model_path)
    # print accuracy of the predictions


