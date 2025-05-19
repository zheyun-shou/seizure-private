from mne.io import read_raw_edf
from mne import make_fixed_length_epochs, Annotations
import pandas as pd
import numpy as np
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_event_info(file_path, epoch_duration):
    """
    Extract event onset, duration, and channel from a BIDS-formatted TSV file.
    
    Returns:
    - events_info (pandas.DataFrame): A DataFrame containing the 'onset', 'duration', and 'channel' columns.
    """
    
    # Load the TSV file using pandas
    events_info = pd.read_csv(file_path, sep='\t')
    
    # Check if the relevant columns exist in the dataframe
    required_columns = ['onset', 'duration', 'channels', 'eventType']
    if not all(col in events_info.columns for col in required_columns):
        raise ValueError(f"One or more of the required columns {required_columns} not found in the file.")
    
    # Drop unnecessary columns
    events_info = events_info.drop(columns=["confidence", "recordingDuration", "dateTime"])

    # Create event info
    events_info.rename({'eventType': 'event_type'}, axis=1, inplace=True)
    # print(events_info)

    return events_info

def read_ids_from_bids(bids_root):
    """
    Read the subject, session, task, and run IDs from the BIDS-formatted directory structure.
    """

    ids = []
    for root, dirs, files in os.walk(bids_root):
        for file in files:
            if file.endswith(".edf"):
                base = file[:-8]  # remove "_eeg.edf" from the end
                json_file = base + "_eeg.json"
                tsv_file = base + "_events.tsv"
                if os.path.exists(os.path.join(root, tsv_file)) and os.path.exists(os.path.join(root, json_file)):
                    rid = get_ids_from_filename(file, return_dict=True)
                    # # create a tree-like dictionary to store the data
                    # rid = {
                    #     'subject_id': subject_id,
                    #     'session_id': session_id,
                    #     'task_id': task_id,
                    #     'run_id': run_id,
                    # }
                    ids.append(rid)
    return ids

def get_ids_from_filename(file_name, return_dict=False):
    """
    Extract subject, session, task, and run IDs from a BIDS-formatted filename.

    """
    
    parts = file_name.split('_')
    subject_id = parts[0].split('-')[-1]
    session_id = parts[1].split('-')[-1]
    task_id = parts[2].split('-')[-1]
    run_id = parts[3].split('-')[-1]
    
    if return_dict:
        return {
            'subject_id': subject_id,
            'session_id': session_id,
            'task_id': task_id,
            'run_id': run_id,
        }
    
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

def extract_epochs(file_path, event_info, downsample=2.0, event_offset=0, epoch_duration=20, epoch_overlap=0, inference=False):
    """
    Extract seizure and nonseizure epochs from a specific recording based on subject, session, task, and run.

    """
    
    def extend_event_info(event_info, event_offset):
        for i, info in event_info.iterrows():
            event_onset = info["onset"] 
            event_duration = info["duration"] + event_offset
            event_end = event_onset + event_duration
            
            # check if out of bound
            if event_onset <= 0:
                event_onset = 0
            elif event_onset + event_duration >= raw_data._last_time:
                event_onset = raw_data._last_time - event_duration if raw_data._last_time - event_duration >= 0 else 0 # move the event onset to accomodate the event duration
            elif event_duration >= raw_data._last_time:
                event_onset = 0
                event_duration = raw_data._last_time
            # update onset, duration and end in event_info
            event_info.at[i, "onset"] = event_onset
            event_info.at[i, "duration"] = event_duration
        return event_info

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
            no_event_intervals.append({"onset": p[0], 
                                       "duration": p[1] - p[0],
                                       "channel": seizure_info.at[0, "channels"],
                                       "event_type": "non_seizure", # non seizure event
                                       }) 
            

        return pd.DataFrame(no_event_intervals)

    def extract_epochs_within_event(extended_event_infos, ids, raw_data, epoch_duration, epoch_overlap):
        epochs = []
        for i, info in extended_event_infos.iterrows():
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
                                                overlap=epoch_overlap, preload=False, verbose=False)
            time_start, time_end, label = [], [], []
            for row in e.events:
                t_start = row[0] / raw_copy.info["sfreq"]
                t_end = row[0] / raw_copy.info["sfreq"] + epoch_duration
                is_seizure = int(any(substring in info["event_type"].lower() for substring in ["sz"]))
                time_start.append(t_start)
                time_end.append(t_end)
                label.append(is_seizure)
            
            # For rows in dataframe, if the row["eventType"] contains str in filter, set row[is_seizure] = 1, else 0
            
            
            e.metadata = pd.DataFrame({"label": label, 
                                       "time_start": time_start, 
                                       "time_end": time_end, 
                                       "subject_id": ids["subject_id"],
                                       "session_id": ids["session_id"],
                                        "task_id": ids["task_id"],
                                        "run_id": ids["run_id"],}) 
            epochs.append(e)
        return epochs

    # Load the EDF file using MNE
    filename = os.path.basename(file_path)
    ids = get_ids_from_filename(filename, return_dict=True)
    raw_data = read_raw_edf(file_path, preload=False, verbose=False)

    # map the channel name from .tsv and .edf
    mapping = {
        ch: ch[:1].upper() + ch[1:2].lower() + ch[2:]
        for ch in raw_data.ch_names
    }
    raw_data.rename_channels(mapping, verbose=False)

    # downsample
    raw_data.resample(
        sfreq=raw_data.info["sfreq"] / downsample,
        method="polyphase",
        verbose=False,
    )

    # Extend the event duration by the event offset,
    seizure_epochs, non_seizure_epochs, bckg_epochs = [], [], []
    if not inference: # extract epochs in order to have a desired ratio
        
        if all(event_info["event_type"] != "bckg"): # info_type == "seizure"
            # seizure_info = extend_event_info(event_info, event_offset)
            seizure_info = event_info
            seizure_epochs = extract_epochs_within_event(seizure_info, ids, raw_data, epoch_duration, epoch_overlap)
            non_seizure_info = get_non_seizure_info(seizure_info, total_duration=raw_data._last_time)
            non_seizure_epochs = extract_epochs_within_event(non_seizure_info, ids, raw_data, epoch_duration, epoch_overlap)
            t = (raw_data._last_time - seizure_info["duration"].sum(), seizure_info["duration"].sum()) # total time of (non_seizure, seizure)
        else: # info_type == "bckg"
            # bckg_info = extend_event_info(event_info, event_offset)
            bckg_info = event_info
            bckg_info["onset"] = 0
            bckg_info["duration"] = raw_data._last_time
            bckg_epochs = extract_epochs_within_event(bckg_info, ids, raw_data, epoch_duration, epoch_overlap)
            t = (raw_data._last_time, 0) # total time of bckg

        return seizure_epochs, non_seizure_epochs, bckg_epochs, t
            
    else: # inference = True, used in evaluation
        epochs = []
        if raw_data._last_time < epoch_duration:
            return epochs
        e = make_fixed_length_epochs(raw_data, 
            duration=epoch_duration,                                    
            overlap=epoch_overlap, 
            verbose=False)
        label = []
        time_start = []
        time_end = []
        for t in e.events:
            t_start = t[0] / raw_data.info["sfreq"]
            t_end = t_start + epoch_duration
            
            is_seizure = 0
            if event_info is not None:
                for i, info in event_info.iterrows():
                    # if the epoch has overlap with seizure, label it as seizure
                    if t_end > info["onset"] and t_start < info["onset"] + info["duration"] and any(substring in info["event_type"].lower() for substring in ["sz"]): 
                        is_seizure = 1
                        break
            label.append(is_seizure)
            time_start.append(t_start)
            time_end.append(t_end)
        e.metadata=pd.DataFrame({"label": label, 
                                "time_start": time_start, 
                                "time_end": time_end, 
                                "subject_id": ids["subject_id"], 
                                "session_id": ids["session_id"],
                                "task_id": ids["task_id"],
                                "run_id": ids["run_id"],})
        epochs.append(e)

        # t = (raw_data._last_time - event_info["duration"].sum(), event_info["duration"].sum()) if event_info is not None else (raw_data._last_time, 0)
        return epochs

def get_data_from_epochs(epochs, return_epoch_data=False):
    segments = []
    for ep in epochs:
        # size of epoch data: n_epochs x n_channels x n_times
        # size of epoch times: n_times, here is 1280
        # size of epoch labels: n_epochs
        epoch = ep.get_data(verbose=False)
        epoch_label = ep.metadata["label"].to_numpy()
        epoch_subject = ep.metadata["subject_id"].to_numpy()
        time_start = ep.metadata["time_start"].to_numpy()
        time_end = ep.metadata["time_end"].to_numpy()
        session_id = ep.metadata["session_id"].to_numpy()
        task_id = ep.metadata["task_id"].to_numpy()
        run_id = ep.metadata["run_id"].to_numpy()
        
        n_epochs, n_channels, n_times = epoch.shape
        
        # epoch = epoch.reshape(n_epochs * n_channels, n_times)
        # epoch_labels = np.tile(ep.metadata["label"], (n_channels))
        segments.append({"epoch": epoch, "label": epoch_label, "subject": epoch_subject, "time_start": time_start, "time_end": time_end, "session_id": session_id, "task_id": task_id, "run_id": run_id})
        
    return segments

def process_recording(ids, bids_root, downsample=2.0, epoch_duration=20, epoch_overlap=0, event_offset=0):
    
    tsv_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="tsv")
    edf_path = get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="edf")
    
    events_info = extract_event_info(tsv_path, epoch_duration)
    
    seizure_epoch, non_seizure_epoch, bckg_epoch, t = extract_epochs(
        edf_path, events_info, downsample, event_offset, epoch_duration, epoch_overlap)
    
    return seizure_epoch, non_seizure_epoch, bckg_epoch, t

def read_dataset(bids_root, subject_ids=None, data_size=1, max_workers=1):
    
    recording_ids = read_ids_from_bids(bids_root)
    # keep the recording ids that are in the subject_ids
    if subject_ids is not None:
        recording_ids = [ids for ids in recording_ids if ids["subject_id"] in subject_ids]
    
    ## save the recording_ids to csv
    # pd.DataFrame(recording_ids).to_csv("train_ids.csv", index=False)
    
    seizure_epochs, non_seizure_epochs, bckg_epochs, ts = [], [], [], []

    if max_workers == 1:
        for ids in recording_ids:
            seizure_epoch, non_seizure_epoch, bckg_epoch, t = process_recording(ids, bids_root, epoch_duration=20, epoch_overlap=0, event_offset=0)
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
    
    num_seizure_epochs  = sum(seizure_epoch_lens) 
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
        return desired_epochs, epoch_cnt, sum(epoch_lens_list)
        
    non_seizure_epochs, non_seizure_epoch_cnt, total_non_seizure_epoch_cnt = get_desired_number_of_epochs(non_seizure_epochs, num_non_seizure_epochs, name="non_seizure")
    bckg_epochs, bckg_epoch_cnt, total_bckg_epoch_cnt = get_desired_number_of_epochs(bckg_epochs, num_bckg_epochs, name="bckg")
    
    # get the data from epochs
    segments = get_data_from_epochs(seizure_epochs)
    segments.extend(get_data_from_epochs(non_seizure_epochs))
    segments.extend(get_data_from_epochs(bckg_epochs))
    
    print(f"Number of segments: {len(segments)}")
    # save the seizure_epoch_lens, non_seizure_epoch_cnt, bckg_epoch_cnt to csv
    df = pd.DataFrame([{"seizure_epochs": num_seizure_epochs, 
                       "total_non_seizure_epoch": total_non_seizure_epoch_cnt,
                       "total_bckg_epoch": total_bckg_epoch_cnt,
                       "non_seizure_epoch_updated": non_seizure_epoch_cnt, 
                       "bckg_epoch_updated": bckg_epoch_cnt}])
    
    
    
    
    return segments, df




# use a sliding windows to first cut the recording into segments
# then label each segment based on the event info





