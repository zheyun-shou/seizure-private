import numpy as np

import matplotlib.pyplot as plt

from scipy.ndimage import generic_filter

from scipy.signal import correlate

import heapq



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
    
    min_max_values = generic_filter(sequence, min_max, size=k, mode='constant', cval=0.0)
    
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

        new_midpoint = max_left + (max_right - max_left) * (midpoint / len(sequence))
        return new_midpoint


    envelope_values = generic_filter(sequence, max_val_envelope, size=k, mode='constant', cval=0.0)
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

        return np.sum(window)
    

    # Apply the sliding window average
    sequence_energy = sequence**2
    energy_values = generic_filter(sequence_energy, energy, size=k, mode='constant', cval=0.0)
    

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
    

    autocorr_values = generic_filter(sequence, autocorr, size=k, mode='constant', cval=0.0)
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



