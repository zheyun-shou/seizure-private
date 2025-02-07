# Example of usage
# # Loading Annotations #

# from timescoring.annotations import Annotation

# # Annotation objects can be instantiated from a binary mask

# fs = 1
# mask = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]

# labels = Annotation(mask, fs)

# print('Annotation objects contain a representation as a mask and as a list of events:')
# print(labels.mask)
# print(labels.events)

# # Annotation object can also be instantiated from a list of events
# fs = 1
# numSamples = 10  # In this case the duration of the recording in samples should be provided
# events = [(1, 3), (6, 9)]

# labels = Annotation(events, fs, numSamples)

# # Computing performance score #

# from timescoring import scoring
# from timescoring import visualization

# fs = 1
# duration = 66 * 60
# ref = Annotation([(8 * 60, 12 * 60), (30 * 60, 35 * 60), (48 * 60, 50 * 60)], fs, duration)
# hyp = Annotation([(8 * 60, 12 * 60), (28 * 60, 32 * 60), (50.5 * 60, 51 * 60), (60 * 60, 62 * 60)], fs, duration)
# scores = scoring.SampleScoring(ref, hyp)
# figSamples = visualization.plotSampleScoring(ref, hyp)

# # Scores can also be computed per event
# param = scoring.EventScoring.Parameters(
#     toleranceStart=30,
#     toleranceEnd=60,
#     minOverlap=0,
#     maxEventDuration=5 * 60,
#     minDurationBetweenEvents=90)
# scores = scoring.EventScoring(ref, hyp, param)
# figEvents = visualization.plotEventScoring(ref, hyp, param)

# print("# Event scoring\n" +
#       "- Sensitivity : {:.2f} \n".format(scores.sensitivity) +
#       "- Precision   : {:.2f} \n".format(scores.precision) +
#       "- F1-score    : {:.2f} \n".format(scores.f1) +
#       "- FP/24h      : {:.2f} \n".format(scores.fpRate))

import joblib
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from mne import make_fixed_length_epochs, Annotations
from timescoring.annotations import Annotation
from timescoring import scoring
from timescoring.scoring import SampleScoring, EventScoring
from timescoring import visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.axes import Axes
import colorsys
import time
from dataloader import extract_epochs, extract_event_info
import torch

def evaluate_recording(edf_path, tsv_path, model_path, downsample=2.0, epoch_duration=10, epoch_overlap=0):
    start_model_time = time.time()

    model = joblib.load(model_path)
    raw_data = read_raw_edf(edf_path, preload=True)
    # desired_channels = raw_data.ch_names[:19]
    # raw_data.pick_channels(desired_channels, verbose=False)
    total_duration = raw_data._last_time
    fs = raw_data.info["sfreq"] / downsample # final sampling freq after downsample

    

    # Build reference Annotation (ref) from .tsv
    
    # - Annotation(events, fs, numSamples)
    # Or by provding a list of (start, stop) tuples for each event. Start and stop
    # times are expected in seconds.

    # The annotation class contains two immutable fields: mask and events.

    # Args:
    #     data (List[Tuple[int, int]] OR NDArray[Bool]): _description_
    #     fs (int): Sampling frequency in Hertz of the annotations.
    #     numSamples (int, optional): Is required when initalizing by providing a
    #         list of (start, stop) tuples. It indicates the number of annotation
    #         samples in the annotation binary mask. It should be left to None if
    #         a binary mask is provided. Defaults to None.

    df_tsv = pd.read_csv(tsv_path, sep='\t')
    ref_events = []
    for _, row in df_tsv.iterrows():
        start_time = int(row['onset'])
        end_time = int(row['onset'] + row['duration'])
        #divide the time into epochs
        # for i in range(start_time, end_time, epoch_duration):
        #     ref_events.append((i, i + epoch_duration))
        ref_events.append((start_time, end_time))

    n_samples = int(total_duration * fs)#?

    ref = Annotation(ref_events, fs, n_samples)


    # Build hypothesis Annotation (hyp) from predicted label=1 epochs
    # raw_data.set_annotations(Annotations(onset=0, duration=total_duration, description="data"))
    event_info = extract_event_info(tsv_path)
    epochs = extract_epochs(edf_path, event_info, downsample)
    segments = []

    # Convert epochs to array for model
    for ep in epochs:
        epoch = ep.get_data() # shape (n_epochs, n_channels, n_times)
        epoch_labels = ep.metadata["label"].to_numpy()
        time_start = ep.metadata["time_start"].to_numpy()
        time_end = ep.metadata["time_end"].to_numpy()
        n_epochs, n_channels, n_times = epoch.shape
        #epoch = epoch.reshape(n_epochs * n_channels, n_times)
        #epoch_labels = np.tile(ep.metadata["label"], (n_channels))

        segments.append({"epoch": epoch, "label": epoch_labels, "time_start": time_start, "time_end": time_end}) 

    X_test = np.concatenate([s["epoch"] for s in segments]).astype(np.float32)
    y_test = np.concatenate([s["label"] for s in segments]).astype(int)
    time_start_test = np.concatenate([s["time_start"] for s in segments])
    time_end_test = np.concatenate([s["time_end"] for s in segments])
    #X_test = X_test[:, np.newaxis, :]
    del segments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_pred = model.predict(X_test)
    end_model_time = time.time()
    print(f"Model prediction took: {end_model_time - start_model_time:.2f} seconds")
    hyp_events = []
    
    for pred, time_start, time_end in zip(y_pred, time_start_test, time_end_test):
        if pred == 1:
            hyp_events.append((time_start, time_end))

    hyp = Annotation(hyp_events, fs, n_samples)
    

    # Compute sample-based scoring
    sample_scores = SampleScoring(ref, hyp)
    figSamples = visualization.plotSampleScoring(ref, hyp)
    print("[Sample-based] Sensitivity:", sample_scores.sensitivity)
    print("[Sample-based] Precision:", sample_scores.precision)
    print("[Sample-based] F1-score:", sample_scores.f1)
    

    # Compute event-based scoring
    param = scoring.EventScoring.Parameters(
    toleranceStart=30,
    toleranceEnd=60,
    minOverlap=0,
    maxEventDuration=5*60,
    minDurationBetweenEvents=90)
    event_scores = scoring.EventScoring(ref, hyp, param)
    figEvents = visualization.plotEventScoring(ref, hyp, param)
    print("[Event-based] Sensitivity:", event_scores.sensitivity)
    print("[Event-based] Precision:", event_scores.precision)
    print("[Event-based] F1-score:", event_scores.f1)

    from analysis import Analyzer

    analyzer = Analyzer(print_conf_mat=True)
    analyzer.analyze_classification(y_pred, y_test, ['normal', 'seizure'])
    accuracy = np.mean(y_pred == y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    plt.show()
    

if __name__ == "__main__":
    edf_path = "E:\BIDS_TUSZ\sub-008\ses-00\eeg\sub-008_ses-00_task-szMonitoring_run-01_eeg.edf"
    tsv_path = "E:\BIDS_TUSZ\sub-008\ses-00\eeg\sub-008_ses-00_task-szMonitoring_run-01_events.tsv"
    model_path = "D:\seizure\models\detach_minirocket_multivariate.pkl"
    evaluate_recording(edf_path, tsv_path, model_path)
