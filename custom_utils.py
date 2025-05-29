import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.signal import correlate
import heapq
from mne.io import read_raw_edf


def get_feature_matrix(segments):
    features = []
    for segment in segments:
        for segment_channel in segment['signal']:
            v = min_max_in_window_scipy(segment_channel, k=64)
            features.append(v)
            # v = get_envelope(segment_channel)
            # v = sliding_average(segment_channel, k=64)
            v = sliding_energy(segment_channel, k=768)
            features.append(v)
            # v = sliding_autocorr(segment_channel, k=768)
            # v = get_custom_envelope(segment_channel, k=768)
    return features


def min_max_in_window_scipy(sequence, k):
    """
    Calculate the maximum absolute value within a sliding window using SciPy for optimized performance.

    Parameters:
    - sequence (np.array): The input sequence of numbers.
    - k (int): The size of the sliding window.

    Returns:
    - max_abs_values (np.array): A list of the maximum absolute values within each sliding window.
    """

    def min_max(window):
        return np.max(window) - np.min(window)

    min_max_values = generic_filter(
        sequence, min_max, size=k, mode='constant', cval=0.0)

    return min_max_values  # Return results for windows fully inside the sequence


def get_custom_envelope(sequence, k):

    def max_val_envelope(sequence):
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

        new_midpoint = max_left + \
            (max_right - max_left) * (midpoint / len(sequence))
        return new_midpoint

    envelope_values = generic_filter(
        sequence, max_val_envelope, size=k, mode='constant', cval=0.0)
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
    def average(window):
        return np.mean(window)

    avg_values = generic_filter(
        sequence, average, size=k, mode='constant', cval=0.0)

    return avg_values  # Return results for windows fully inside the sequence


def sliding_energy(sequence, k):
    """
    Calculate the sliding average of a sequence using a window size of k.

    Parameters:
    - sequence (np.array): The input sequence of numbers.
    - k (int): The size of the sliding window.

    Returns:
    - avg_values (np.array): A list of the average values within each sliding window.
    """

    def energy(window):
        return np.sum(window)

    sequence_energy = sequence**2
    energy_values = generic_filter(
        sequence_energy, energy, size=k, mode='constant', cval=0.0)

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

    def autocorr(window):
        # # normalize the window by subtracting the mean and min-max scaling
        # window = (window - np.mean(window)) / (np.max(window) - np.min(window))
        # arr = np.correlate(window, window, mode='same')
        # return np.max(arr)
        arr = correlate(window, window, mode='full')
        arr = arr[arr.size//2:]

        # find the first zero-crossing
        def first_zero_crossing_point(arr):
            idx = np.where(np.diff(np.sign(arr)) != 0)[0]
            if idx.size > 0:
                return 1e-4
                # return arr[idx[0] + 1]
            return 0

        return first_zero_crossing_point(arr)

    autocorr_values = generic_filter(
        sequence, autocorr, size=k, mode='constant', cval=0.0)
    return autocorr_values


def get_labels_from_info(times, event_info):
    """
    Get the labels for each sample in the segment based on the event information.

    Parameters:
    - times (np.array): The times for each sample in the segment.
    - event_info (pd.DataFrame): The event information DataFrame.

    Returns:
    - labels (np.array): The labels for each sample in the segment.
    """

    labels = np.zeros(len(times))
    for _, event in event_info.iterrows():
        onset_event = event['onset']
        duration_event = event['duration']
        end_time_event = onset_event + duration_event
        # Find the indices of the samples that fall within the event time window
        event_indices = np.where((times >= onset_event) & (times < end_time_event))[0]
        # Assign the event label to the corresponding samples
        labels[event_indices] = 1

    return labels

def print_confusion_matrix(ConfMat, label_strings=None, title='Confusion matrix'):
    """Print confusion matrix as text to terminal"""

    if label_strings is None:
        label_strings = ConfMat.shape[0] * ['']

    print(title)
    print(len(title) * '-')
    # Make printable matrix:
    print_mat = []
    for i, row in enumerate(ConfMat):
        print_mat.append([label_strings[i]] + list(row))
    print(tabulate(print_mat, headers=['True\Pred'] + label_strings, tablefmt='orgtbl'))

def plot_confusion_matrix(ConfMat, label_strings=None, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    """Plot confusion matrix in a separate window"""
    plt.imshow(ConfMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if label_strings:
        tick_marks = np.arange(len(label_strings))
        plt.xticks(tick_marks, label_strings, rotation=90)
        plt.yticks(tick_marks, label_strings)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
    plt.plot(times, segment_data.T)
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
    
# write a function to plot certain channel, certain time window of a recording
def plot_eeg_channel_recording(file_path, channel_name, start_time, end_time):
    """
    Plots a specific channel segment of an EEG recording.

    Parameters:
    - file_path (str): Path to the EEG file.
    - channel_name (str): Name of the channel to plot.
    - start_time (float): Start time of the segment in seconds.
    - end_time (float): End time of the segment in seconds.
    """
    raw = read_raw_edf(file_path, preload=False)
    raw.pick_channels([channel_name])
    
    # Get data and times
    data, times = raw[channel_name]
    
    # Convert times to seconds
    times = times / raw.info['sfreq']
    
    # Filter data for the specified time window
    mask = (times >= start_time) & (times <= end_time)
    
    plt.figure(figsize=(10, 4))
    plt.plot(times[mask], data[0][mask])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'EEG Segment for Channel {channel_name} from {start_time}s to {end_time}s')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = 'F:/BIDS_Siena/sub-00/ses-01/eeg/sub-00_ses-01_task-szMonitoring_run-00_eeg.edf'
    channel_name = 'T4-Avg'
    start_time = 1100
    end_time = 1300
    plot_eeg_channel_recording(file_path, channel_name, start_time, end_time)