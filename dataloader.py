from bids import BIDSLayout
# import mne
from mne.io import read_raw_edf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.signal import envelope
from wavelet_utils import wavelet_decompose_channels_from_segment

def extract_segment(subject, session, task, run, channel, start_time, duration, bids_root):
    """
    Extract a segment from a specific recording based on subject, session, task, and run.

    """
    
    # Construct the file path
    file_name = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_eeg.edf"
    file_path = f"{bids_root}/sub-{subject}/ses-{session}/eeg/{file_name}"
    
    # Load the EDF file using MNE
    raw_data = read_raw_edf(file_path, preload=True)
    
    # Check if the specified channel is in the data
    # if channel not in raw_data.info['ch_names']:
    #     raise ValueError(f"Channel '{channel}' not found in the data.")
    
    # Find the index of the channel
    # channel_idx = raw_data.info['ch_names'].index(channel)
    raw_data.pick_channels(channel)
    
    # Convert start time and duration to sample indices
    sfreq = raw_data.info['sfreq']
    start_sample = int(start_time * sfreq)
    end_sample = int((start_time + duration) * sfreq)
    
    # Extract the desired segment from the data
    data, times = raw_data.get_data(return_times=True)
    segment_data = data[:, start_sample:end_sample]
    segment_times = times[start_sample:end_sample]
    return segment_data, segment_times

def extract_event_info(subject, session, task, run, path_to_tsv_files):
    """
    Extract event onset, duration, and channel from a BIDS-formatted TSV file.
    
    Returns:
    - events_df (pandas.DataFrame): A DataFrame containing the 'onset', 'duration', and 'channel' columns.
    """
    
    # Construct the file path
    file_name = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
    file_path = f"{bids_root}/sub-{subject}/ses-{session}/eeg/{file_name}"
    
    # Load the TSV file using pandas
    events_df = pd.read_csv(file_path, sep='\t')
    
    # Check if the relevant columns exist in the dataframe
    required_columns = ['onset', 'duration', 'channels']
    if not all(col in events_df.columns for col in required_columns):
        raise ValueError(f"One or more of the required columns {required_columns} not found in the file.")
    
    # Extract the relevant columns: onset, duration, and channel
    events_info = events_df[['onset', 'duration', 'channels']]
    
    return events_info

def plot_eeg_segment(segment_data, times, channel_name, event_info):
    """
    Plots a segment of EEG data.

    Parameters:
    - segment_data (ndarray): Time-series data for the segment.
    - times (ndarray): Time values corresponding to the segment.
    - channel_name (str): Name of the channel being plotted.
    """
    #highlight the seizure events
    # for event in seizure_events:
     

    plt.figure(figsize=(10, 4))
    plt.plot(times, segment.T)
    for _, event in event_info.iterrows():
        onset_event = event['onset']
        duration_event = event['duration']
        end_time_event = onset_event + duration_event
        if end_time_event > times[-1] or onset_event < times[0]:
            continue
        plt.axvspan(onset_event, end_time_event, color='green', alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'EEG Segment for Channel {channel_name}')
    plt.grid(True)
    # plt.show()

def roi_overlap_ratio(start_time_roi, end_time_roi, event_info):
    """
    Calculate the overlap ratio of an extracted ROI with reference to ground truth events.
    
    Parameters:
    - start_time_roi (float): The start time of the ROI (in seconds).
    - end_time_roi (float): The end time of the ROI (in seconds).
    - event_info (pandas.DataFrame): DataFrame containing 'onset', 'duration', and 'channel' of events.
    
    Returns:
    - overlap_ratios (list): List of overlap ratios for each event.
    """
    
    overlap_ratios = []
    
    for _, event in event_info.iterrows():
        onset_event = event['onset']
        duration_event = event['duration']
        end_time_event = onset_event + duration_event
        
        # Calculate overlap times
        overlap_start = max(start_time_roi, onset_event)
        overlap_end = min(end_time_roi, end_time_event)
        
        # If there is an overlap, calculate the overlap duration
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
        else:
            overlap_duration = 0
        
        # Calculate the overlap ratio
        event_duration = duration_event
        if event_duration > 0:
            overlap_ratio = overlap_duration / event_duration
        else:
            overlap_ratio = 0
        
        overlap_ratios.append(overlap_ratio)
    
    return overlap_ratios



def min_max_in_window_scipy(sequence, k):
    """
    Calculate the maximum absolute value within a sliding window using SciPy for optimized performance.
    
    Parameters:
    - sequence (np.array): The input sequence of numbers.
    - k (int): The size of the sliding window.
    
    Returns:
    - max_abs_values (np.array): A list of the maximum absolute values within each sliding window.
    """
    
    # Define a custom function for max absolute value in the window
    # def max_abs(window):
    #     return np.max(np.abs(window))
    # Define a custom function for min max value in the window
    def min_max(window):
        return np.max(window) - np.min(window)
    
    # Apply the sliding window max absolute value
    max_abs_values = generic_filter(sequence, min_max, size=k, mode='constant', cval=0.0)
    
    return max_abs_values  # Return results for windows fully inside the sequence

def get_envelope(sequence):
    bp_in = (256*4, None)
    x_env, x_res = envelope(sequence, bp_in=bp_in)
    return x_env + x_res

def get_custom_envelope(sequence, k):

    def custom_envelope(sequence):
        # 1. find the midpoint of the sequence
        # 2. find the max value to the left and right of the midpoint
        # 3. linear interpolation between the two max values
        # find the midpoint
        midpoint = len(sequence) // 2
        # find the max value to the left of the midpoint
        max_left = np.max(sequence[:midpoint])
        # find the max value to the right of the midpoint
        max_right = np.max(sequence[midpoint:])
        # get new midpoint value using linear interpolation
        new_midpoint = max_left + (max_right - max_left) * (midpoint / len(sequence))
        return new_midpoint

    envelope_values = generic_filter(sequence, custom_envelope, size=k, mode='constant', cval=0.0)
    return envelope_values

def sliding_average(sequence, k):
    """
    Calculate the sliding average of a sequence using a window size of k.
    
    Parameters:
    - sequence (np.array): The input sequence of numbers.
    - k (int): The size of the sliding window.
    
    Returns:
    - avg_values (np.array): A list of the average values within each sliding window.
    """
    
    # Define a custom function for average in the window
    def average(window):
        return np.mean(window)
    
    # Apply the sliding window average
    avg_values = generic_filter(sequence, average, size=k, mode='constant', cval=0.0)
    
    return avg_values  # Return results for windows fully inside the sequence

# def compute_power_spectral_density(segment_data, sfreq):

    
#     from scipy.signal import welch
    
#     # Compute the PSD using Welch's method
#     freqs, psd = welch(segment_data, sfreq, nperseg=256)
    
#     return freqs, psd

def sliding_energy(sequence, k):
    """
    Calculate the sliding average of a sequence using a window size of k.
    
    Parameters:
    - sequence (np.array): The input sequence of numbers.
    - k (int): The size of the sliding window.
    
    Returns:
    - avg_values (np.array): A list of the average values within each sliding window.
    """
    
    # Define a custom function for average in the window
    def energy(window):
        return np.sum(window**2)
    
    # Apply the sliding window average
    energy_values = generic_filter(sequence, energy, size=k, mode='constant', cval=0.0)
    
    return energy_values

def sliding_autocorr(sequence, k):
    """
    Calculate the sliding average of a sequence using a window size of k.
    
    Parameters:
    - sequence (np.array): The input sequence of numbers.
    - k (int): The size of the sliding window.
    
    Returns:
    - avg_values (np.array): A list of the average values within each sliding window.
    """
    
    # Define a custom function for autocorrelation in the window, and clip to the right size in the middle
    def autocorr(window):
        # normalize the window by subtracting the mean and min-max scaling
        window = (window - np.mean(window)) / (np.max(window) - np.min(window))
        arr = np.correlate(window, window, mode='same')
        return np.max(arr)
    
    # Apply the sliding window average
    autocorr_values = generic_filter(sequence, autocorr, size=k, mode='constant', cval=0.0)
    
    return autocorr_values

bids_root = 'E:\BIDS_Siena' # Replace with your actual path
task = 'szMonitoring'                # Task name
subject = '01'                       # Subject ID
session = '01'                       # Session ID
run = '00'                           # Run ID
# desired_channel = ['T3-Avg', 'T5-Avg']            # Channel name
desired_channel = [] 
start_time = 46253                  # Start time in seconds
duration = 300.0                      # Duration in seconds


segment, times = extract_segment(subject, session, task, run, desired_channel, start_time, duration, bids_root)

event_info = extract_event_info(subject, session, task, run, bids_root)
print(event_info)

# decom_wavelets = wavelet_decompose_channels_from_segment(segment, times, desired_channel, event_info, level=5, output=True)

# Plot the segment
plot_eeg_segment(segment, times, desired_channel, event_info)

# max values
for segment_channel in segment:
    # v = min_max_in_window_scipy(segment_channel, k=64)
    # v = get_envelope(segment_channel)
    # v = sliding_average(segment_channel, k=64)
    # v= sliding_energy(segment_channel, k=768)
    # v = sliding_autocorr(segment_channel, k=768)
    # v = get_custom_envelope(segment_channel, k=768)
    # scale v to the same max/min as the segment_channel
    # v = v * (np.max(segment_channel) - np.min(segment_channel)) / (np.max(v) - np.min(v))
    plt.plot(times, v)
plt.show()

# Calculate the overlap ratio
overlap_ratios = roi_overlap_ratio(start_time, start_time + duration, event_info)

