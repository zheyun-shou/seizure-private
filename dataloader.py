from mne.io import read_raw_edf
import pandas as pd
import os

def extract_segment(file_path, segment_info):
    """
    Extract a segment from a specific recording based on subject, session, task, and run.

    """

    # Load the EDF file using MNE
    raw_data = read_raw_edf(file_path, preload=True)

    channels, start_time, duration = segment_info["channel"], segment_info["start_time"], segment_info["duration"]
    mapping = {
        ch: ch[:1].upper() + ch[1:2].lower() + ch[2:]
        for ch in raw_data.ch_names
    }
    raw_data.rename_channels(mapping)

    # Check if the specified channel is in the data
    for ch in channels:
        if ch not in raw_data.info['ch_names']:
            raise ValueError(f"Channel '{ch}' not found in the data.")
    
    # Find the index of the channel
    raw_data.pick(channels)
    
    # Convert start time and duration to sample indices
    sfreq = raw_data.info['sfreq']
    start_sample = int(start_time * sfreq)
    end_sample = int((start_time + duration) * sfreq)
    
    # Extract the desired segment from the data
    data, times = raw_data.get_data(return_times=True)
    segment_data = data[:, start_sample:end_sample]
    segment_times = times[start_sample:end_sample]
    return segment_data, segment_times

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

def get_channel_from_event_info(event_info, offset=100):
    chs = (event_info["channels"].to_list()[0]).split(sep=",")
    for i in range(len(chs)):
        chs[i] = chs[i] + "-Avg"
    onset = event_info["onset"].to_list()[0]
    duration = event_info["duration"].to_list()[0]
    onset -= offset
    duration += 2 * offset
    info = {"channel": chs, "start_time": onset, "duration": duration}
    return info

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

def read_siena_dataset(bids_root):
    recording_ids = read_ids_from_bids(bids_root)
    event_infos, segments = [], []
    for ids in recording_ids:
        event_info = extract_event_info(get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="tsv"))
        segment_info = get_channel_from_event_info(event_info)
        signal, time = extract_segment(get_path_from_ids(ids, bids_root, get_abs_path=True, file_format="edf"), segment_info)
        event_infos.append(event_info)
        segments.append({"signal": signal, "time": time})
    return event_infos, segments



# use a sliding windows to first cut the recording into segments
# then label each segment based on the event info





