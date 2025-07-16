from datetime import datetime
import json
import os
import mne
import pandas as pd
import numpy as np
import random
import psutil
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Helper to extract BIDS ids from filename
import re

from tqdm import tqdm
import yaml

def extract_bids_ids_from_path(path):
    # Example: sub-458_ses-00_task-szMonitoring_run-01_events.tsv
    fname = os.path.basename(path)
    m = re.match(r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_]+)_events.tsv", fname)
    if not m:
        raise ValueError(f"Filename does not match BIDS pattern: {fname}")
    return m.groupdict()

def extract_path_from_bids_ids(ids, bids_root=None, get_abs_path=False, format="edf"):
    suffix = "eeg" if (format == "edf" or format == "json") else "events"
    subject, session, task, run = ids['subject'], ids['session'], ids['task'], ids['run']
    _dirname = f"{bids_root}/sub-{subject}/ses-{session}/eeg/"
    _filename = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_{suffix}.{format}"
    if get_abs_path:
        assert bids_root is not None, "Please provide the BIDS root directory."
        return os.path.join(_dirname, _filename)
    else:
        return _filename

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    sys.exit(0)

def get_system_memory_usage():
    """Get system-wide memory usage in GB"""
    try:
        memory = psutil.virtual_memory()
        return memory.used / 1073741824  # Convert to GB
    except Exception:
        return 0.0

def is_seizure_event(event_row, config):
    return any(substring in event_row['eventType'].lower() for substring in config['seizure_event_type'])

def read_all_events(config: dict):
    """
    Gather all events from *_events.tsv files, label intervals as 'sz', 'interictal', or 'bckg',
    and append BIDS ids to each row. Returns a concatenated DataFrame.
    """
    all_rows = []
    epoch_duration = config['epoch_duration']
    if not os.path.exists(config['bids_root']):
        raise FileNotFoundError(f"BIDS root directory not found: {config['bids_root']}")
    for root, _, files in os.walk(config['bids_root']):
        files.sort()
        for file in files:
            if file.endswith('_events.tsv'):
                tsv_path = os.path.join(root, file)
                ids = extract_bids_ids_from_path(tsv_path)
                df = pd.read_csv(tsv_path, sep='\t')
                # Get recording duration (assume all rows have same value)
                rec_duration = df['recordingDuration'].iloc[0]
                # valid check: 
                # 1. onset should be > 0; otherwise reset to 0;
                # 2. onset + duration should be <= recordingDuration, otherwise reset to recordingDuration
                # condition: recording duration should align with recording.last_time
                df['onset'] = df['onset'].apply(lambda x: 0 if x < 0 else x)
                df['duration'] = df.apply(lambda row: rec_duration - row['onset'] if row['duration'] + row['onset'] > rec_duration else row['duration'], axis=1)
                # Sort by onset
                df = df.sort_values('onset').reset_index(drop=True)
                # If any seizure exists, fill gaps with interictal
                is_seizure = [is_seizure_event(row, config) for _, row in df.iterrows()]
                if any(is_seizure): #seizure recording
                    intervals = []
                    last_end = 0.0
                    for i, row in df.iterrows():
                        extended_onset = (row['onset'] // epoch_duration) * epoch_duration
                        extended_end = np.ceil((row['onset'] + row['duration']) / epoch_duration) * epoch_duration
                        df.loc[i, 'onset'] = extended_onset
                        df.loc[i, 'end'] = extended_end if extended_end <= rec_duration else extended_end - epoch_duration
                        df.loc[i, 'duration'] = df.loc[i, 'end'] - df.loc[i, 'onset']

                    for i, row in df.iterrows():
                        if row['onset'] > last_end:
                            # gap before this event
                            intervals.append({
                                'onset': last_end,
                                'duration': row['onset'] - last_end,
                                'end': row['onset'],
                                'eventType': 'interictal',
                                'recordingDuration': rec_duration
                            })
                    
                        intervals.append({
                            'onset': row['onset'],
                            'duration': row['duration'],
                            'end': row['onset'] + row['duration'],
                            'eventType': 'sz',
                            'recordingDuration': rec_duration
                        })
                        last_end = row['onset'] + row['duration']

                    if last_end < rec_duration:
                        intervals.append({
                            'onset': last_end,
                            'duration': rec_duration - last_end,
                            'end': rec_duration,
                            'eventType': 'interictal',
                            'recordingDuration': rec_duration
                        })
                    out_df = pd.DataFrame(intervals)
                else: #bckg recording
    
                    out_df = pd.DataFrame([{
                        'onset': 0.0,
                        'duration': rec_duration,
                        'end': rec_duration,
                        'eventType': 'bckg',
                        'recordingDuration': rec_duration
                    }])

                # Add ids
                for k, v in ids.items():
                    out_df[k] = v
                all_rows.append(out_df)

    if all_rows:
        result = pd.concat(all_rows, ignore_index=True)
        # check for each subject if there is at least one seizure recording, and append the bool value to the result
        unique_subjects = result['subject'].unique()
        seizure_subjects = []
        for subject in unique_subjects:
            event_in_subject_idx = (result['subject'] == subject)
            subject_df = result[event_in_subject_idx]
            is_seizure_subject = False
            for _, row in subject_df.iterrows():
                if is_seizure_event(row, config):
                    is_seizure_subject = True
                    seizure_subjects.append(subject)
                    break
            result.loc[event_in_subject_idx, 'is_seizure_subject'] = is_seizure_subject
        # print how many subjects are seizure subjects and how many are not
        bckg_subjects = [subject for subject in unique_subjects if subject not in seizure_subjects]
        print(f"Number of seizure subjects: {len(seizure_subjects)}")
        print(f"Number of bckg subjects: {len(bckg_subjects)}")
        result = result.drop_duplicates()
        return result, (seizure_subjects, bckg_subjects)
    else:
        return pd.DataFrame()

def create_balanced_epochs(df, config):
    """
    Create balanced fixed-length epochs from labeled events.
    
    Args:
        df: DataFrame with labeled events (output from read_all_events)
        config: Configuration dictionary
        
    Returns:
        tuple: (list of epoch dictionaries, statistics dictionary)
    """
    epoch_duration = config['epoch_duration']
    ratios = config['epoch_ratios']
    debug = config['debug']

    print(df["duration"].sum())

    epoch_dfs = []
    for i, row in df.iterrows():
        # rules for creating epoch dataframes:
        # 1. all the epochs are aligned to the start of the recording, with the duration of epoch_duration
        # 2. for is_seizure_event = True, we do over-estimation, i.e. contain the epoch if any time point is in the seizure event
        # 3. for eventType = non_seizure, we only sample within the interval & aligned with n times of epoch_duration
        # left align & seizure events close to each other with less than epoch_duration will be merged into one event
        epoch_slices = np.arange(row['onset'], row['end'] + 1e-6, epoch_duration)
        epoch_start_times = epoch_slices[:-1]
        epoch_end_times = epoch_slices[1:]
        # if is_seizure_event(row, config):
        #     if epoch_slices[-1] > row['end'] and epoch_slices[-1] + epoch_duration < row['recordingDuration']:
        #         epoch_start_times = np.concatenate([epoch_start_times, [epoch_slices[-1]]])
        #         epoch_end_times = np.concatenate([epoch_end_times, [epoch_slices[-1] + epoch_duration]])
        #     if epoch_slices[0] < row['onset'] and epoch_slices[0] - epoch_duration > 0:
        #         epoch_start_times = np.concatenate([[epoch_slices[0] - epoch_duration], epoch_start_times])
                # epoch_end_times = np.concatenate([[epoch_slices[0]], epoch_end_times])
        
        # create a epoch dataframe with the start and end times
        epoch_df = pd.DataFrame({
            'onset': epoch_start_times,
            'duration': epoch_duration,
            'end': epoch_end_times,
            'eventType': row['eventType'],
            'subject': row['subject'],
            'session': row['session'],
            'task': row['task'],
            'run': row['run'],
            'recordingDuration': row['recordingDuration'],
            'is_seizure_subject': row['is_seizure_subject']
        })
        epoch_dfs.append(epoch_df)
    all_epoch_df = pd.concat(epoch_dfs, ignore_index=True)
    # Calculate total durations
    durations_by_type = all_epoch_df.groupby('eventType')['duration'].sum().to_dict()
    ref_type = min(durations_by_type, key=durations_by_type.get)
    num_epochs_by_type = {k: int(ratios[k] / ratios[ref_type] * durations_by_type[ref_type] / epoch_duration) for k, v in durations_by_type.items()}
    if debug:
        print(f"Total durations: {durations_by_type}")
        print(f"Total epochs: {num_epochs_by_type}")
    
    # TODO：data ratio: e.g. 50%, 80%, subject-wise 
    
    # Sample epochs from each event type
    all_epochs = []

    # random sample the epochs from each event type
    for event_type, num_epochs in num_epochs_by_type.items():
        event_type_epochs = all_epoch_df[all_epoch_df['eventType'] == event_type]
        event_type_epochs = event_type_epochs.sample(n=num_epochs, random_state=config['sample_seed'], replace=True)
        all_epochs.append(event_type_epochs)
    all_epochs = pd.concat(all_epochs, ignore_index=True)

    # keep only unique epochs
    all_epochs = all_epochs.drop_duplicates()
    return all_epochs.to_dict(orient='records') 

def load_multi_epochs_from_recording(recording_data):
    """
    Load multiple epochs from a single recording - this function will be called by worker processes.
    
    Args:
        recording_data: tuple of (recording_epochs_list, config_dict, recording_index)
        
    Returns:
        tuple: (recording_index, success, results_list) or (recording_index, success, error_message, None)
    """
    recording_epochs, config, recording_index = recording_data
    
    try:
        # Configure MNE verbosity for this worker process
        mne.set_log_level('ERROR' if not config['mne_verbose'] else 'INFO')
        
        # Get the first epoch to extract recording info (all epochs in this batch have same recording)
        first_epoch = recording_epochs[0]
        ids = {
            'subject': first_epoch['subject'],
            'session': first_epoch['session'],
            'task': first_epoch['task'],
            'run': first_epoch['run']
        }
        
        # Load the EDF file once for all epochs in this recording
        edf_file = extract_path_from_bids_ids(ids, bids_root=config['bids_root'], get_abs_path=True, format="edf")
        
        # Load the EDF file using mne library
        edf_raw = mne.io.read_raw_edf(
            edf_file, 
            preload=config['mne_preload'], 
            verbose=config['mne_verbose'])
        
        # Downsample the data once
        edf_raw.resample(
            sfreq=config['downsample'],
            verbose=config['mne_verbose'],
            )
        
        # Process all epochs from this recording
        results = []
        for epoch in recording_epochs:
            try:
                # Get the data from the epoch
                start = int(epoch['onset'] * edf_raw.info['sfreq'])
                stop = int(epoch['duration'] * edf_raw.info['sfreq']) + start

                data = edf_raw.get_data(
                    start=start, 
                    stop=stop,
                    verbose=config['mne_verbose'],
                    )
                
                # Assign the label to the epoch
                epoch_x = data
                epoch_y = 1 if is_seizure_event(epoch, config) else 0
                
                results.append((epoch_x, epoch_y))
                
            except Exception as e:
                print(f"Error processing epoch in recording {recording_index}: {str(e)}")
                # Continue with other epochs in this recording
                continue
        
        return (recording_index, True, results)
        
    except Exception as e:
        return (recording_index, False, str(e), None)

def group_epochs_by_recording(epochs):
    """
    Group epochs by their recording (subject, session, task, run).
    
    Args:
        epochs: List of epoch dictionaries
        
    Returns:
        dict: Dictionary with recording key as (subject, session, task, run) and value as list of epochs
    """
    grouped = {}
    for epoch in epochs:
        key = (epoch['subject'], epoch['session'], epoch['task'], epoch['run'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(epoch)
    return grouped

def find_matching_config(config, dir):
    """ Check if a config file exists that matches the config, return the config file if found """
    config_files = [f for f in os.listdir(dir) if f.endswith('.yaml')]
    for config_file in config_files:
        with open(os.path.join(dir, config_file), 'r') as f:
            config_from_file = yaml.safe_load(f)
            if config_from_file['data_size'] == config['data_size'] and \
                config_from_file['split_ratio'] == config['split_ratio'] and \
                config_from_file['epoch_duration'] == config['epoch_duration'] and \
                config_from_file['downsample'] == config['downsample'] and \
                config_from_file['epoch_ratios'] == config['epoch_ratios'] and \
                config_from_file['sample_seed'] == config['sample_seed'] and \
                config_from_file['split_seed'] == config['split_seed'] and \
                config_from_file['num_kernels'] == config['num_kernels'] and \
                config_from_file['num_models'] == config['num_models'] and \
                config_from_file['model_type'] == config['model_type']:
                print(f"Found matching config file: {config_file}")
                return config_from_file
    return None

def load_epoch_data(epochs, config, split_name):
    """ Load epoch data from edf file using mne library and assign labels with multiprocessing support """
    X, y = [], []
    
    assert split_name in ['train', 'test'], "Split name must be either 'train' or 'test'"

    if config.get('load_preprocessed_data', False):
        # 1. scan the preprocessed_dir for config files
        # 2. load the config file
        # 3. check the following terms if they are the same as the config:
        #    - split_ratio
        #    - epoch_duration
        #    - downsample
        #    - epoch_ratios
        #    - rnd_seed
        # 4. if all terms are the same, load the preprocessed data
        # 5. if any term is different, load the epochs and preprocess the data

        config_from_file = find_matching_config(config, config['preprocessed_dir'])
        if config_from_file is not None:
            print(f"Loading preprocessed data...")
            timestamp = config_from_file['timestamp']
            data_file = os.path.join(config['preprocessed_dir'], f'data_{split_name}_{timestamp}.npz')
            if os.path.exists(data_file):
                data = np.load(data_file)
                return data['X'], data['y']
            else:
                print(f"Data file not found: {data_file}")
        
        # if no matching config file is found, load the epochs and preprocess the data
        print(f"No matching config file found for split {split_name}. Loading epochs and preprocessing the data...")
    
    # Get initial memory usage
    initial_memory = get_system_memory_usage()
    
    # Set up signal handling for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    max_workers = config.get('max_workers', 1)
    
    if max_workers == 1:
        # Single-threaded processing
        pbar = tqdm(epochs, desc="Loading epoch data", ncols=100, leave=True)
        
        for i, epoch in enumerate(pbar):
            # Update progress bar with memory usage
            current_memory = get_system_memory_usage()
            memory_diff = current_memory - initial_memory
            pbar.set_postfix({
                'Memory': f'{current_memory:.1f}GB',
            })
            
            try:
                # load the edf file
                ids = {
                    'subject': epoch['subject'],
                    'session': epoch['session'],
                    'task': epoch['task'],
                    'run': epoch['run']
                }
                edf_file = extract_path_from_bids_ids(ids, bids_root=config['bids_root'], get_abs_path=True, format="edf")
                
                # load the edf file using mne library
                edf_raw = mne.io.read_raw_edf(
                    edf_file, 
                    preload=config['mne_preload'], 
                    verbose=config['mne_verbose'])
                
                # downsample the data
                edf_raw.resample(
                    sfreq=config['downsample'],
                    verbose=config['mne_verbose'],
                    )
                
                # get the data from the epoch
                # ensure the start and stop are integers, and duration is edf_raw.info['sfreq'] * epoch_duration
                start = int(epoch['onset'] * edf_raw.info['sfreq'])
                stop = int(epoch['duration'] * edf_raw.info['sfreq']) + start

                data = edf_raw.get_data(
                    start=start, 
                    stop=stop,
                    verbose=config['mne_verbose'],
                    )
                
                # assign the label to the epoch
                epoch_x = data
                epoch_y = 1 if epoch['eventType'] == 'sz' else 0
                X.append(epoch_x)
                y.append(epoch_y)
                
            except Exception as e:
                print(f"\nError processing epoch {i}: {e}")
                print(f"Epoch details: {epoch}")
                continue
    else:
        # Multiprocessing with batch processing by recording
        # Group epochs by recording
        grouped_epochs = group_epochs_by_recording(epochs)
        
        # Create a mapping from recording key back to epoch indices for ordering
        epoch_to_recording = {}
        recording_to_epochs = {}

        print(f"\nStarting to load {len(epochs)} epochs using {max_workers} workers...")
        
        for i, epoch in enumerate(epochs):
            key = (epoch['subject'], epoch['session'], epoch['task'], epoch['run'])
            epoch_to_recording[i] = key
            if key not in recording_to_epochs:
                recording_to_epochs[key] = []
            recording_to_epochs[key].append(i)
        
        # Prepare data for multiprocessing (one recording per task)
        recording_data_list = []
        for recording_index, (recording_key, recording_epochs) in enumerate(grouped_epochs.items()):
            recording_data_list.append((recording_epochs, config, recording_index))
        
        # Initialize result arrays
        results = [None] * len(epochs)
        print("\nLoading recordings...")
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all recording tasks
                future_to_recording = {executor.submit(load_multi_epochs_from_recording, recording_data): recording_data[2] 
                                     for recording_data in recording_data_list}
                
                # Process completed tasks with progress bar
                pbar = tqdm(as_completed(future_to_recording), total=len(future_to_recording), 
                           desc="", ncols=100)
                
                for future in pbar:
                    recording_index = future_to_recording[future]
                    
                    try:
                        result = future.result()
                        recording_index, success, data_or_error = result
                        
                        if success:
                            # data_or_error is now a list of (data, label) tuples
                            epoch_results = data_or_error
                            
                            # Get the recording key for this recording_index
                            recording_key = list(grouped_epochs.keys())[recording_index]
                            
                            # Map results back to original epoch indices
                            epoch_indices = recording_to_epochs[recording_key]
                            for i, (data, label) in enumerate(epoch_results):
                                if i < len(epoch_indices):
                                    results[epoch_indices[i]] = (data, label)
                        else:
                            print(f"\nError processing recording {recording_index}: {data_or_error}")
                        
                        # Update progress bar with system memory usage
                        current_memory = get_system_memory_usage()
                        successful_epochs = sum(1 for r in results if r is not None)
                        pbar.set_postfix({
                            'Mem': f'{current_memory:.1f}GB',
                            'Success': f'{successful_epochs}/{len(results)}'
                        })
                        
                    except Exception as e:
                        print(f"\nUnexpected error processing recording {recording_index}: {str(e)}")
                        print(f"  Error type: {type(e).__name__}")
                        import traceback
                        print(f"  Traceback: {traceback.format_exc()}")
                        continue
                        
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Shutting down gracefully...")
            # Cancel all pending tasks
            for future in future_to_recording:
                future.cancel()
            print("All pending tasks cancelled.")
            return X, y
        
        # Collect results in order
        for i, result in enumerate(results):
            if result is not None:
                data, label = result
                X.append(data)
                y.append(label)

    print(f"Successfully loaded {len(X)} epochs")

    X = np.array(X)
    y = np.array(y)

    if config.get('save_preprocessed_data', False):
        # Save both X and y in a single npz file
        timestamp = config['timestamp']
        np.savez_compressed(
            os.path.join(config['preprocessed_dir'], f'data_{split_name}_{timestamp}.npz'), 
            X=X, 
            y=y
        )
        # config is the same for all splits, so save it once
        with open(os.path.join(config['preprocessed_dir'], f'config_{timestamp}.yaml'), 'w') as f:
            yaml.dump(config, f)
        print(f"Preprocessed data saved to: {config['preprocessed_dir']}")
    
    return X, y

def train_test_epochs_split(epochs, config):
    """
    Split epochs into train and test epochs.
    """
    split_ratio = config['split_ratio']
    random_seed = config['rnd_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)

    # group epochs
    grouped_epochs = group_epochs_by_recording(epochs)
    print(f"Grouped into {len(grouped_epochs)} recordings")

    # find unique combinations of subjects, sessions, tasks, runs
    unique_combinations = set()
    for recording_key, recording_epochs in grouped_epochs.items():
        unique_combinations.add(recording_key)
    print(f"Found {len(unique_combinations)} unique combinations of subjects, sessions, tasks, runs")

    # split unique combinations into train and test
    unique_combinations_list = list(unique_combinations)
    random.shuffle(unique_combinations_list)
    
    split_index = int(len(unique_combinations_list) * split_ratio)
    train_combinations = set(unique_combinations_list[:split_index])
    test_combinations = set(unique_combinations_list[split_index:])
    
    print(f"Train combinations: {len(train_combinations)}, Test combinations: {len(test_combinations)}")

    # save the train and test combinations to a json file
    timestamp = config['timestamp']
    with open(os.path.join(config['preprocessed_dir'], f'train_test_split_{timestamp}.json'), 'w') as f:
        json.dump({
            'train': list(train_combinations),
            'test': list(test_combinations)
        }, f)
    
    # Convert epochs to DataFrame for easier manipulation
    epochs_df = pd.DataFrame(epochs)
    
    # Create a key column for grouping
    epochs_df['recording_key'] = epochs_df.apply(
        lambda row: (row['subject'], row['session'], row['task'], row['run']), axis=1
    )
    
    # Split epochs based on recording combinations
    train_epochs = epochs_df[epochs_df['recording_key'].isin(train_combinations)].copy()
    test_epochs = epochs_df[epochs_df['recording_key'].isin(test_combinations)].copy()
    
    # Remove the temporary key column
    train_epochs = train_epochs.drop('recording_key', axis=1)
    test_epochs = test_epochs.drop('recording_key', axis=1)

    
    print(f"Train epochs: {len(train_epochs)}, Test epochs: {len(test_epochs)}")
    
    return train_epochs, test_epochs
    
def load_all_epochs_from_recording(recording_data):
    """
    Load all epochs from a single recording - this function will be called by worker processes.
    """
    recording_ids, config = recording_data
    
    try:
        # Configure MNE verbosity for this worker process
        mne.set_log_level('ERROR' if not config['mne_verbose'] else 'INFO')
        
        # Load the EDF file once for all epochs in this recording
        edf_file = extract_path_from_bids_ids(recording_ids, bids_root=config['bids_root'], get_abs_path=True, format="edf")
        tsv_file = extract_path_from_bids_ids(recording_ids, bids_root=config['bids_root'], get_abs_path=True, format="tsv")

        # Load the events file
        rec_event = pd.read_csv(tsv_file, sep='\t')
        rec_event = rec_event.sort_values('onset').reset_index(drop=True)

        # Load the EDF file using mne library
        edf_raw = mne.io.read_raw_edf(
            edf_file, 
            preload=config['mne_preload'], 
            verbose=config['mne_verbose'])
        
        fs = edf_raw.info["sfreq"] / config['downsample']
        
        # Downsample the data once
        edf_raw.resample(
            sfreq=config['downsample'],
            verbose=config['mne_verbose'],
            )
        
        # make fixed length epochs
        epochs, labels, interval = [], [], []
        if edf_raw._last_time < config['epoch_duration']:
            return recording_ids, epochs, labels, interval, rec_event, fs
        
        e = mne.make_fixed_length_epochs(edf_raw, 
            duration=config['epoch_duration'],
            preload=config['mne_preload'],
            verbose=config['mne_verbose'])

        for row in e.events:
            t_start = row[0] / edf_raw.info["sfreq"]
            t_end = t_start + config['epoch_duration']
            
            is_seizure_epoch = 0
            for i, info in rec_event.iterrows():
                # if the epoch has overlap with seizure, label it as seizure
                if is_seizure_event(info, config) and (t_end > info["onset"]) and (t_start < info["onset"] + info["duration"]): 
                    is_seizure_epoch = 1
                    break
            labels.append(is_seizure_epoch)
            interval.append((t_start, t_end))
        
        # Get the actual epoch data
        epoch_data = e.get_data(verbose=config['mne_verbose'])
        
        return recording_ids, epoch_data, labels, interval, rec_event, fs
    
    except Exception as e:
        print(f"Error loading all epochs from recording {recording_ids}: {str(e)}")
        return recording_ids, None, None, None, None, None

def load_evaluation_data_from_recording(config: dict):
    """
    Load evaluation data from test recordings with multiprocessing support.
    This loads complete recordings (not balanced samples) for evaluation.
    """
    # Load the train_test_split
    with open(os.path.join(config['preprocessed_dir'], f'train_test_split_{config["timestamp"]}.json'), 'r') as f:
        train_test_split = json.load(f)

    # Get the test recording IDs
    test_combinations = train_test_split['test']
    print(f"Loading evaluation data from {len(test_combinations)} test recordings...")
    
    # Convert combinations back to recording IDs format
    test_recording_ids = []
    for subject, session, task, run in test_combinations:
        test_recording_ids.append({
            'subject': subject,
            'session': session,
            'task': task,
            'run': run
        })
    
    # Get initial memory usage
    initial_memory = get_system_memory_usage()
    
    # Set up signal handling for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    max_workers = config.get('max_workers', 1)
    
    if max_workers == 1:
        # Single-threaded processing
        results = {}
        pbar = tqdm(test_recording_ids, desc="Loading evaluation recordings", ncols=100)
        
        for recording_ids in pbar:
            # Update progress bar with memory usage
            current_memory = get_system_memory_usage()
            pbar.set_postfix({
                'Memory': f'{current_memory:.1f}GB',
                'Loaded': f'{len(results)}/{len(test_recording_ids)}'
            })
            
            try:
                recording_ids, epoch_data, labels, interval, rec_event, fs = load_all_epochs_from_recording((recording_ids, config))
                if epoch_data is not None:
                    results[tuple(recording_ids.values())] = {
                        'epoch_data': epoch_data,
                        'labels': labels,
                        'interval': interval,
                        'events': rec_event,
                        'recording_ids': recording_ids,
                        'fs': fs
                    }
            except Exception as e:
                print(f"\nError processing recording {recording_ids}: {e}")
                continue
    else:
        # Multiprocessing
        print(f"Using {max_workers} workers for parallel evaluation data loading...")
        
        # Prepare data for multiprocessing
        recording_data_list = [(recording_ids, config) for recording_ids in test_recording_ids]
        
        # Initialize results dictionary
        results = {}
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all recording tasks
                future_to_recording = {executor.submit(load_all_epochs_from_recording, recording_data): recording_data[0] 
                                     for recording_data in recording_data_list}
                
                # Process completed tasks with progress bar
                pbar = tqdm(as_completed(future_to_recording), total=len(future_to_recording), 
                           desc="Loading evaluation recordings", ncols=100)
                
                for future in pbar:
                    recording_ids = future_to_recording[future]
                    
                    try:
                        result = future.result()
                        recording_ids, epoch_data, labels, interval, rec_event, fs = result
                        
                        if epoch_data is not None:
                            results[tuple(recording_ids.values())] = {
                                'epoch_data': epoch_data,
                                'labels': labels,
                                'interval': interval,
                                'events': rec_event,
                                'recording_ids': recording_ids,
                                'fs': fs
                            }
                        
                        # Update progress bar with system memory usage
                        current_memory = get_system_memory_usage()
                        pbar.set_postfix({
                            'Mem': f'{current_memory:.1f}GB',
                            'Loaded': f'{len(results)}/{len(test_recording_ids)}'
                        })
                        
                    except Exception as e:
                        print(f"\nUnexpected error processing recording {recording_ids}: {str(e)}")
                        print(f"  Error type: {type(e).__name__}")
                        import traceback
                        print(f"  Traceback: {traceback.format_exc()}")
                        continue
                        
        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Shutting down gracefully...")
            # Cancel all pending tasks
            for future in future_to_recording:
                future.cancel()
            print("All pending tasks cancelled.")
            return results
    
    print(f"\nEvaluation data loading completed:")
    
    # Print summary statistics
    total_epochs = sum(len(data['epoch_data']) for data in results.values())
    total_seizure_epochs = sum(sum(data['labels']) for data in results.values())
    print(f"Total epochs: {total_epochs}, Seizure epochs: {total_seizure_epochs} ({total_seizure_epochs/total_epochs*100:.1f}%)")
    
    return results

def analyze_recording_performance(recording_results, model_predictions):
    """
    Analyze performance metrics for each recording.
    
    Args:
        recording_results: Dictionary from load_evaluation_data_from_recording
        model_predictions: Dictionary with same keys as recording_results, containing predictions
        
    Returns:
        DataFrame with performance metrics per recording
    """
    analysis_results = []
    
    for recording_key, data in recording_results.items():
        true_labels = data['labels']
        pred_labels = model_predictions[recording_key]
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division='0'
        )
        
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        
        # Calculate additional metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        analysis_results.append({
            'recording_key': recording_key,
            'total_epochs': len(true_labels),
            'seizure_epochs': sum(true_labels),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        })
    
    return pd.DataFrame(analysis_results)
