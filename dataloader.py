from mne.io import read_raw_edf
from mne import make_fixed_length_epochs, Annotations
import pandas as pd
import numpy as np
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_event_info(file_path):
    """
    Extract event onset, duration, and channel from a BIDS-formatted TSV file.
    
    Returns:
    - events_df (pandas.DataFrame): A DataFrame containing the 'onset', 'duration', and 'channel' columns.
    """
    
    # Load the TSV file using pandas
    events_df = pd.read_csv(file_path, sep='\t')
    
    # Check if the relevant columns exist in the dataframe
    required_columns = ['onset', 'duration', 'channels']
    if not all(col in events_df.columns for col in required_columns):
        raise ValueError(f"One or more of the required columns {required_columns} not found in the file.")
    
    # Extract the relevant columns: onset, duration, and channel
    events_info = events_df[['onset', 'duration', 'channels']]
    
    return events_info

def read_ids_from_bids(bids_root):
    """
    Read the subject, session, task, and run IDs from the BIDS-formatted directory structure.
    """

    recording_ids = []
    for root, dirs, files in os.walk(bids_root):
        for file in files:
            if file.endswith(".edf"):
                base = file[:-8]  # remove "_eeg.edf" from the end
                json_file = base + "_eeg.json"
                tsv_file = base + "_events.tsv"
                if os.path.exists(os.path.join(root, tsv_file)) and os.path.exists(os.path.join(root, json_file)):
                    subject_id, session_id, task_id, run_id = get_ids_from_filename(file)
                    # create a tree-like dictionary to store the data
                    ids = {
                        'subject_id': subject_id,
                        'session_id': session_id,
                        'task_id': task_id,
                        'run_id': run_id,
                    }
                    recording_ids.append(ids)
    return recording_ids

def get_ids_from_filename(file_name):
    """
    Extract subject, session, task, and run IDs from a BIDS-formatted filename.

    """
    
    parts = file_name.split('_')
    subject_id = parts[0].split('-')[-1]
    session_id = parts[1].split('-')[-1]
    task_id = parts[2].split('-')[-1]
    run_id = parts[3].split('-')[-1]
    
    return subject_id, session_id, task_id, run_id

def get_dir_from_ids(bids_root, ids):
    subject_id, session_id = ids['subject_id'], ids['session_id']
    return f"{bids_root}/sub-{subject_id}/ses-{session_id}/eeg/"

def get_path_from_ids(ids, bids_root=None, get_abs_path=False, file_format="edf"):
    suffix = "eeg"if (file_format == "edf" or file_format == "json") else "events"
    subject_id, session_id, task_id, run_id = ids['subject_id'], ids['session_id'], ids['task_id'], ids['run_id']
    data_file = f"sub-{subject_id}_ses-{session_id}_task-{task_id}_run-{run_id}_{suffix}.{file_format}"
    if get_abs_path:
        assert bids_root is not None, "Please provide the BIDS root directory."
        return os.path.join(get_dir_from_ids(bids_root, ids), data_file)
    else:
        return data_file
    
def get_channel_from_event_info(event_info):
    chs = (event_info["channels"].to_list()[0]).split(sep=",")
    for i in range(len(chs)):
        chs[i] = chs[i] + "-Avg"
    return chs

def sample_non_seizure_intervals(event_infos, total_duration=0, n_samples=5):
    """
    total_duration: the total duration of the recording
    """
    no_event_intervals = []
    # if total_duration > 0:
    #     longest_seizure_duration = total_duration / n_samples
    # else:
    #     longest_seizure_duration = event_infos["duration"].max()
    
    seizure_intervals = list(zip(event_infos["onset"], event_infos["duration"]))

    # Define non-seizure periods
    no_event_periods = []
    if seizure_intervals[0][0] > 0: # if the first seizure does not start at time 0
        no_event_periods.append((0, seizure_intervals[0][0]))
    for i in range(len(seizure_intervals) - 1):
        no_event_periods.append((seizure_intervals[i][0]+seizure_intervals[i][1], seizure_intervals[i + 1][0]))
    if seizure_intervals[-1][0]+seizure_intervals[-1][1] < total_duration:
        no_event_periods.append((seizure_intervals[-1][0]+seizure_intervals[-1][1], total_duration))

    # Randomly sample from non-seizure periods
    # 1. has the proportinal (n_samples=10) length as seizure
    # 2. (future) have same epoch length as seizure(eg. 10s)

    sum_seizure_duration = event_infos["duration"].sum()
    for _ in range(n_samples):
        while no_event_periods:
            selected_period = no_event_periods[np.random.randint(len(no_event_periods))]
            period_duration = selected_period[1] - selected_period[0]
            if period_duration >= sum_seizure_duration:
                onset = np.random.uniform(selected_period[0], selected_period[1] - sum_seizure_duration)
                no_event_intervals.append({"onset": onset, "duration": sum_seizure_duration})
                break
            else:
                no_event_periods.remove(selected_period)

    return pd.DataFrame(no_event_intervals)

def extract_epochs(file_path, event_info, downsample=2.0, event_offset=100, epoch_duration=10, epoch_overlap=5):
    """
    Extract seizure and nonseizure epochs from a specific recording based on subject, session, task, and run.

    """

    # Load the EDF file using MNE
    raw_data = read_raw_edf(file_path, preload=True)

    # map the channel name from .tsv and .edf
    mapping = {
        ch: ch[:1].upper() + ch[1:2].lower() + ch[2:]
        for ch in raw_data.ch_names
    }
    raw_data.rename_channels(mapping)

    # Check if the specified channel is in the data
    desired_channels = []
    # desired_channels = get_channel_from_event_info(event_info)
    # for ch in desired_channels:
    #     if ch not in raw_data.info['ch_names']:
    #         raise ValueError(f"Channel '{ch}' not found in the data.")
    if desired_channels:
        raw_data.pick_channels(desired_channels, verbose=False)
    # raw_data.pick(desired_channels, verbose=False)
    # downsample to 128Hz
    raw_data.resample(
        sfreq=raw_data.info["sfreq"] / downsample,
        method="polyphase",
        verbose=False,
    )
    
    # Extend the event duration by the event offset,
    updated_event_infos = []
    for i, info in event_info.iterrows():

        event_duration = info["duration"] + event_offset
        event_onset = info["onset"] 
        
        # check if out of bound
        if event_onset < 0:
            event_onset = 0
        elif event_onset + event_duration > raw_data._last_time:
            event_onset = raw_data._last_time - event_duration # if too long, move the onset to accomodate
        
        updated_event_info = {"channel": desired_channels, "onset": event_onset, "duration": event_duration}
        updated_event_infos.append(updated_event_info)

        raw_data.set_annotations(annotations=Annotations(onset=event_onset, duration=event_duration, description="seizure"))

    updated_event_infos = pd.DataFrame(updated_event_infos)

    no_event_info = sample_non_seizure_intervals(updated_event_infos, total_duration=raw_data._last_time, n_samples=1)

    epochs, labels = [], []
    for i, info in updated_event_infos.iterrows():
        raw_copy = raw_data.copy().crop(info["onset"], info["onset"] + info["duration"], verbose=False)
        # create epochs
        e = make_fixed_length_epochs(raw_copy, duration=epoch_duration,
                                            overlap=epoch_overlap, preload=True)
        label = []
        for t in e.events:
            t_start = t[0] / raw_copy.info["sfreq"]
            t_end = (t[0] + epoch_duration) / raw_copy.info["sfreq"]
            # if the epoch has overlap with seizure, label it as seizure
            if t_end > event_info["onset"][i] and t_start < event_info["onset"][i] + event_info["duration"][i] : 
                label.append(1)
            else:
                label.append(0)
         # label the epochs, 1: seizure, 0: non-seizure       
        e.metadata=pd.DataFrame({"label": label})
        epochs.append(e)


    for i, info in no_event_info.iterrows():
        raw_copy = raw_data.copy().crop(info["onset"], info["onset"] + info["duration"], verbose=False)
        e = make_fixed_length_epochs(raw_copy, duration=epoch_duration,
                                            overlap=epoch_overlap, preload=True, verbose=False)
        label = [0] * len(e.events)
        e.metadata=pd.DataFrame({"label": label})
        epochs.append(e)

    return epochs

def process_recording(ids, bids_root, downsample=2.0, epoch_duration=10, epoch_overlap=5, event_offset=100):
    event_info = extract_event_info(get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="tsv"))
    epochs = extract_epochs(get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="edf"), 
                            event_info, downsample, event_offset, epoch_duration, epoch_overlap)
    segments = []
    for ep in epochs:
        # size of epoch data: n_epochs x n_channels x n_times
        # size of epoch times: n_times, here is 1280
        # size of epoch labels: n_epochs
        epoch = ep.get_data()
        epoch_labels = ep.metadata["label"].to_numpy()
        
        n_epochs, n_channels, n_times = epoch.shape
        
        epoch = epoch.reshape(n_epochs * n_channels, n_times)
        epoch_labels = np.tile(ep.metadata["label"], (n_channels))
        
        segments.append({"epoch": epoch, "label": epoch_labels})
        # TODO: Add properties that allows trace back to the original recording
    return segments

def read_siena_dataset(bids_root, max_workers=4):
    recording_ids = read_ids_from_bids(bids_root)
    event_infos, segments = [], []

    if max_workers == 1:
        for ids in recording_ids:
            segment = process_recording(ids, bids_root, epoch_duration=10, epoch_overlap=5, event_offset=100)
            segments.extend(segment)
    else:
        # multirpocessing in cpu
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_recording, ids, bids_root) for ids in recording_ids]
            for future in as_completed(futures):
                segments.extend(future.result())

    return event_infos, segments




# use a sliding windows to first cut the recording into segments
# then label each segment based on the event info





