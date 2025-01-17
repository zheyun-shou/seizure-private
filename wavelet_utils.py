# ...existing code...
import re  # Added for regex operations if not already imported
import pandas as pd  # Ensure pandas is imported

def wavelet_decompose_channels_from_segment(segment, times, desired_channel, event_info=None, level=5, output=False):
    """
    Modified wavelet decomposition method to handle `segment`, `times`, `desired_channel`, and `event_info`
    from dataloader.py.
    
    segment: numpy array of shape (n_channels, n_samples)
    times: numpy array of shape (n_samples,)
    desired_channel: list of channel names
    event_info: not directly used here, but included for future expansions
    """
    # Convert `segment` to a DataFrame where each channel is a column
    # segment shape is (channels, samples), so transpose for (samples, channels)
    df = pd.DataFrame(segment.T, columns=desired_channel)
    df.columns.name = 'channel'
    
    # # Example downsampling step:
    # df = df.iloc[0::2, :]  # take every other sample

    # Transpose again so wavelet is done per channel
    df_t = df.transpose()
    
    from pywt import wavedec
    
    # Perform wavelet decomposition
    # Example usage: wavelet='db4'
    # Example output: [cA5, cD5, cD4, cD3, cD2, cD1]
    val_t = df_t.to_numpy()
    # print(val_t.shape)
    coeffs_list = wavedec(val_t, wavelet='db4', level=level)
    # print(coeffs_list)
    
    # Prepare names for detail (D) and approximation (A) levels
    nums = list(range(1, level+1))
    names = [f'D{i}' for i in nums] + [f'A{nums[-1]}']
    names = names[::-1]
    
    wavelets = pd.DataFrame()
    for i, array in enumerate(coeffs_list):
        level_df = pd.DataFrame(array)
        level_df.index = df_t.index
        level_df['level'] = names[i]
        level_df = level_df.set_index('level', append=True)
        level_df = level_df.T
        wavelets = pd.concat([wavelets, level_df], axis=1, sort=True)
    
    # Sort multi-index columns (channel, level)
    wavelets = wavelets.sort_values(['channel', 'level'], axis=1)
    
    # Remove approximation levels if only details needed
    regex = re.compile('D')
    bad_items = [x for x in list(wavelets.columns.levels[1]) if not regex.match(x)]
    decom_wavelets = wavelets.drop(bad_items, axis=1, level='level')
    
    decom_wavelets.index.name = 'sample'
    
    if output:
        print(decom_wavelets.head())
    
    return decom_wavelets


def power_measure_channels(data, freq, output=False):
  welch_df = pd.DataFrame()
  for channel_name in data:
    channel_df = pd.DataFrame(power_measures(pd.DataFrame(data[channel_name])))
    channel_df['channel'] = channel_name
    channel_df.index.name = 'feature'
    channel_df = channel_df.set_index('channel', append=True)
    channel_df = channel_df.swaplevel()

    if welch_df.empty:
        welch_df = channel_df
    else:
        welch_df = pd.concat([welch_df, channel_df])

  welch_df = welch_df.T

  if output:
    print(welch_df.head())
    
  return welch_df

