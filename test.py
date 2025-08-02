# import re


# if __name__ == "__main__":
#     regExToFind = r"^(EEG )?([A-Z]{1,2}[1-9]*)(-[a-z]?[1-9]*)?"
#     channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
#     print(len(channels))
#     for i, channel in enumerate(channels):
#         result = re.search(regExToFind, channel, flags=re.IGNORECASE)
#         if result.group(2) is not None:
#             electrode = result.group(2)
#         else:
#             electrode = channel
#         channels[i] = "{}-{}".format(electrode, "REF")

#     print(len(channels))
#     print(channels)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

T_x, N = 1 / 20, 1000  # 20 Hz sampling rate for 50 s signal
t_x = np.arange(N) * T_x  # time indexes for signal
f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # varying frequency
x = np.sin(2*np.pi*np.cumsum(f_i)*T_x) 

g_std = 8  # standard deviation for Gaussian window in samples
w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
SFT = ShortTimeFFT(w, hop=10, fs=1/T_x, mfft=200, scale_to='magnitude')
Sx = SFT.stft(x)  # perform the STFT

print(Sx.shape)

