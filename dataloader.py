from mne.io import read_raw_edf
from mne import make_fixed_length_epochs, Annotations
import pandas as pd
import numpy as np
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_event_info(file_path, epoch_duration=10, filter=["sz", "seiz"]):
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
    valid_idx = []
    for i, row in events_df.iterrows():
        # check if the row["eventType"] contains elements in filter
        if not any(f in row["eventType"].lower() for f in filter):
            continue
        valid_idx.append(i)
    events_info = events_df[['onset', 'duration', 'channels']]
    
    if len(valid_idx) > 0:
        return events_info.loc[valid_idx, :]
    else:
        return None

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

def extract_epochs(file_path, event_info, downsample=2.0, event_offset=0, epoch_duration=10, epoch_overlap=0, info_type="seizure", inference=False):
    """
    Extract seizure and nonseizure epochs from a specific recording based on subject, session, task, and run.

    """
    
    def extend_event_info(event_info, event_offset):
        
        extended_event_infos = []
        for i, info in event_info.iterrows():
            
            event_duration = info["duration"] + event_offset
            event_onset = info["onset"] 
            
            # check if out of bound
            if event_onset <= 0:
                event_onset = 0
            elif event_onset + event_duration >= raw_data._last_time:
                event_onset = raw_data._last_time - event_duration if raw_data._last_time - event_duration >= 0 else 0 # move the event onset to accomodate the event duration
            elif event_duration >= raw_data._last_time:
                event_onset = 0
                event_duration = raw_data._last_time

            extended_event_infos.append({"channel": desired_channels, "onset": event_onset, "duration": event_duration})

        extended_event_infos = pd.DataFrame(extended_event_infos)
        return extended_event_infos

    def get_non_seizure_info(seizure_info, total_duration):
        no_event_intervals = []
        
        seizure_intervals = list(zip(seizure_info["onset"], seizure_info["duration"]))

        # Define non-seizure periods
        no_event_periods = []
        if seizure_intervals[0][0] > 0: # if the first seizure does not start at time 0
            no_event_periods.append((0, seizure_intervals[0][0]))
        for i in range(len(seizure_intervals) - 1):
            no_event_periods.append((seizure_intervals[i][0]+seizure_intervals[i][1], seizure_intervals[i + 1][0]))
        if seizure_intervals[-1][0]+seizure_intervals[-1][1] < total_duration:
            no_event_periods.append((seizure_intervals[-1][0]+seizure_intervals[-1][1], total_duration))
                    
        for p in no_event_periods:
            no_event_intervals.append({"onset": p[0], "duration": p[1] - p[0]})

        return pd.DataFrame(no_event_intervals)

    # Load the EDF file using MNE
    # extract filename from filepath
    filename = os.path.basename(file_path)
    subject_id, session_id, task_id, run_id = get_ids_from_filename(filename)
    raw_data = read_raw_edf(file_path, preload=True, verbose=False)
    # map the channel name from .tsv and .edf
    mapping = {
        ch: ch[:1].upper() + ch[1:2].lower() + ch[2:]
        for ch in raw_data.ch_names
    }
    raw_data.rename_channels(mapping, verbose=False)
    # Check if the specified channel is in the data
    desired_channels = []
    if desired_channels:
        raw_data.pick_channels(desired_channels, verbose=False)
    # downsample
    raw_data.resample(
        sfreq=raw_data.info["sfreq"] / downsample,
        method="polyphase",
        verbose=False,
    )
    
    # Extend the event duration by the event offset,
    seizure_epochs, non_seizure_epochs, epochs = [], [], []
    if not inference:
        
        
        if info_type == "seizure":
            updated_seizure_info = extend_event_info(event_info, event_offset)
            for i, info in updated_seizure_info.iterrows():
                # add duration length check to make sure at least one epoch is available
                duration = max(info["duration"], epoch_duration) 
                start = max(0, info["onset"])
                end = min(info["onset"] + duration, raw_data._last_time)
                # skip if cannot create a valid epoch
                if start >= end or end - start < duration:
                    continue
                # create epochs
                raw_copy = raw_data.copy().crop(start, end, verbose=False)
                e = make_fixed_length_epochs(raw_copy, duration=epoch_duration,
                                                    overlap=epoch_overlap, preload=True, verbose=False)
                label, time_start, time_end = [], [], []
                for t in e.events:
                    t_start = t[0] / raw_copy.info["sfreq"]
                    t_end = t[0] / raw_copy.info["sfreq"] + epoch_duration
                    # if the epoch has overlap with seizure, label it as seizure
                    if t_end > event_info["onset"][i] and t_start < event_info["onset"][i] + event_info["duration"][i] : 
                        label.append(1)
                    else:
                        label.append(0)
                    time_start.append(t_start)
                    time_end.append(t_end)
                
                e.metadata=pd.DataFrame({"label": label, "time_start": time_start, "time_end": time_end, "subject_id":subject_id})
                seizure_epochs.append(e)
            updated_non_seizure_info = get_non_seizure_info(event_info, total_duration=raw_data._last_time)
            t = (raw_data._last_time - event_info["duration"].sum(), event_info["duration"].sum()) # total time of (non_seizure, seizure)
        else: # info_type == "bckg"
            updated_non_seizure_info = extend_event_info(event_info, event_offset) #convert format
            updated_non_seizure_info["onset"] = 0
            updated_non_seizure_info["duration"] = raw_data._last_time
            t = (raw_data._last_time, 0) # total time of bckg
        
        for i, info in updated_non_seizure_info.iterrows():
            # add duration length check to make sure at least one epoch is available
            duration = max(info["duration"], epoch_duration) 
            start = max(0, info["onset"])
            end = min(info["onset"] + info["duration"], raw_data._last_time)
            # skip if cannot create a valid epoch
            if start >= end or end - start < duration:
                continue
            raw_copy = raw_data.copy().crop(start, end, verbose=False)
            e = make_fixed_length_epochs(raw_copy, duration=epoch_duration,
                                                overlap=epoch_overlap, preload=True, verbose=False)
            time_start = [t[0] / raw_copy.info["sfreq"] for t in e.events]
            time_end = [t[0] / raw_copy.info["sfreq"] + epoch_duration for t in e.events]
            label = [0] * len(e.events)
            e.metadata = pd.DataFrame({"label": label, "time_start": time_start, "time_end": time_end, "subject_id":subject_id})
            non_seizure_epochs.append(e)
        
        return seizure_epochs, non_seizure_epochs, t
            
    else: # inference = True, used in evaluation
       
        if raw_data._last_time < epoch_duration:
            return epochs
        e = make_fixed_length_epochs(raw_data, duration=epoch_duration,
                                                overlap=epoch_overlap, preload=True)
        label = []
        time_start = []
        time_end = []
        for t in e.events:
            t_start = t[0] / raw_data.info["sfreq"]
            t_end = t[0] / raw_data.info["sfreq"] + epoch_duration
            
            is_seizure = 0
            if event_info is not None:
                for i, info in event_info.iterrows():
                    # if the epoch has overlap with seizure, label it as seizure
                    if t_end > info["onset"] and t_start < info["onset"] + info["duration"]: 
                        is_seizure = 1
                        break
            label.append(is_seizure)
            time_start.append(t_start)
            time_end.append(t_end)
        e.metadata=pd.DataFrame({"label": label, "time_start": time_start, "time_end": time_end, "subject_id":subject_id})
        epochs.append(e)

        t = (raw_data._last_time - event_info["duration"].sum(), event_info["duration"].sum()) if event_info is not None else (raw_data._last_time, 0)
        return epochs, t

def get_data_from_epochs(epochs):
    segments = []
    for ep in epochs:
        # size of epoch data: n_epochs x n_channels x n_times
        # size of epoch times: n_times, here is 1280
        # size of epoch labels: n_epochs
        epoch = ep.get_data()
        epoch_label = ep.metadata["label"].to_numpy()
        epoch_subject = ep.metadata["subject_id"].to_numpy()
        
        n_epochs, n_channels, n_times = epoch.shape
        
        # epoch = epoch.reshape(n_epochs * n_channels, n_times)
        # epoch_labels = np.tile(ep.metadata["label"], (n_channels))
        segments.append({"epoch": epoch, "label": epoch_label, "subject": epoch_subject})
        
    return segments

def process_recording(ids, bids_root, downsample=2.0, epoch_duration=10, epoch_overlap=0, event_offset=0):
    
    tsv_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="tsv")
    edf_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="edf")
    
    seizure_info = extract_event_info(tsv_path, epoch_duration, filter=["sz", "seiz"])
    bckg_info = extract_event_info(tsv_path, epoch_duration, filter=["bckg"])
    
    seizure_epoch, non_seizure_epoch, bckg_epoch, t = None, None, None, None
    
    if seizure_info is not None:
        seizure_epoch, non_seizure_epoch, t = extract_epochs(
            edf_path, seizure_info, downsample, event_offset, epoch_duration, epoch_overlap, info_type="seizure")
    
    elif bckg_info is not None:
        _, bckg_epoch, t = extract_epochs(
            edf_path, bckg_info, downsample, event_offset, epoch_duration, epoch_overlap, info_type="bckg")
    
    return seizure_epoch, non_seizure_epoch, bckg_epoch, t

def read_siena_dataset(bids_root, max_workers=1):
    recording_ids = read_ids_from_bids(bids_root)
    # keep the recordings that has an odd number of subject id. TODO: remove this line
    recording_ids = [ids for ids in recording_ids if int(ids["subject_id"]) % 2 == 1]
    # save the recording_ids to csv
    pd.DataFrame(recording_ids).to_csv("train_ids.csv", index=False)
    
    seizure_epochs, non_seizure_epochs, bckg_epochs, ts = [], [], [], []

    if max_workers == 1:
        for ids in recording_ids:
            seizure_epoch, non_seizure_epoch, bckg_epoch, t = process_recording(ids, bids_root, epoch_duration=10, epoch_overlap=0, event_offset=0)
            if seizure_epoch is not None:
                seizure_epochs.extend(seizure_epoch)
            if non_seizure_epoch is not None:
                non_seizure_epochs.extend(non_seizure_epoch)
            if bckg_epoch is not None:
                bckg_epochs.extend(bckg_epoch)
            ts.append(t)
    else:
        # multirpocessing in cpu
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_recording, ids, bids_root) for ids in recording_ids]
            for future in as_completed(futures):
                res = future.result()
                if res[0] is not None:
                    seizure_epochs.extend(res[0])
                if res[1] is not None:
                    non_seizure_epochs.extend(res[1])
                if res[2] is not None:
                    bckg_epochs.extend(res[2])
                ts.append(res[3])
                    
    # save ts into .csv file
    pd.DataFrame(ts, columns=["non_seizure_length", "seizure_length"]).to_csv("ts.csv", index=False)
    del ts
    
                    
    print(f"Number of recordings: {len(recording_ids)}")
    print(f"Number of subjects: {len(np.unique([ids['subject_id'] for ids in recording_ids]))}")                
    
    # random select index in non_seizure_epochs and bckg_epochs
    import random
    random.seed(0)

    seizure_epoch_lens = [e.events.shape[0] for e in seizure_epochs]
    print("num_seizure_epochs:", sum(seizure_epoch_lens))
    
    num_non_seizure_epochs = sum(seizure_epoch_lens)
    num_bckg_epochs = sum(seizure_epoch_lens) // 2

    # shuffle the idx for non_seizure_epochs and bckg_epochs
    def get_desired_number_of_epochs(epochs, desired_num_epochs, name="non_seizure"):
        import random
        random.seed(0)
        
        # target number of epochs for non_seizure and bckg
        epoch_lens_list = [e.events.shape[0] for e in epochs] # number of epochs in each recording
        print(f"num_{name}_epochs:", sum(epoch_lens_list))
        desired_num_epochs = min(int(desired_num_epochs), sum(epoch_lens_list))
        
        idx = list(range(len(epochs)))
        random.shuffle(idx)
        
        epoch_cnt = 0
        desired_epochs = []
        for i in idx:
            if epoch_cnt >= desired_num_epochs:
                break
            desired_epochs.append(epochs[i])
            epoch_cnt += epoch_lens_list[i]
        print(f"num_{name}_epochs, updated:", epoch_cnt)
        return desired_epochs
        
    non_seizure_epochs = get_desired_number_of_epochs(non_seizure_epochs, num_non_seizure_epochs, name="non_seizure")
    bckg_epochs = get_desired_number_of_epochs(bckg_epochs, num_bckg_epochs, name="bckg")
    
    # get the data from epochs
    segments = get_data_from_epochs(seizure_epochs)
    segments.extend(get_data_from_epochs(non_seizure_epochs))
    segments.extend(get_data_from_epochs(bckg_epochs))
    
    return segments




# use a sliding windows to first cut the recording into segments
# then label each segment based on the event info





