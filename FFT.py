import numpy as np
from scipy import signal
from scipy.signal import stft
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import os
from sklearn.model_selection import StratifiedKFold
from mne_utils import setup_mne_for_processing
from new_dataloader import load_epoch_data, read_all_events, create_balanced_epochs
from scipy.signal.windows import gaussian, hann
from scipy.signal import ShortTimeFFT

def power_spectral_density_feature(signal, samplerate, new_length=None):

    freq_resolution = 2 # higher the better resolution but time consuming...
    def psd(amp, begin, end, freq_resol = freq_resolution):
        return np.average(amp[begin*freq_resol:end*freq_resol], axis=0)

    nperseg = 8
    # noverlap = 4
    noverlap = 0
    nfft = samplerate * freq_resolution
    freqs, times, spec = stft(signal, samplerate, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=True, boundary='zeros')
    amp = (np.log(np.abs(spec) + 1e-10))

    # new_length = int(new_length)
    # if abs(amp.shape[1] - new_length) > 1:
    #     print("Difference is huge {} {}".format(amp.shape[1], new_length))
    # amp = amp[:,:new_length]

    # new_sig = sci_sig.resample(signal, len(times))
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(signal)    
    # plt.subplot(3, 1, 2)
    # plt.plot(new_sig)    
    # plt.subplot(3, 1, 3)
    # plt.pcolormesh(amp)
    # plt.show()

    # print("freqs: ", freqs)
    # print("times: ", times)
    # print("freqs: ", len(freqs))
    # print("times: ", len(times))
    # print("signal len: ", len(signal))
    # exit(1)

    psds = []
    if samplerate == 128:
        psd1 = psd(amp,0,4)
        psd2 = psd(amp,4,8)
        psd3 = psd(amp,8,13)
        psd4 = psd(amp,13,20)
        psd5 = psd(amp,20,30)
        psd6 = psd(amp,30,40)
        psd7 = psd(amp,40,60)

        psds = [psd1, psd2, psd3, psd4, psd5, psd6, psd7]

    if samplerate == 256:
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.641.3620&rep=rep1&type=pdf
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8910555
        psd1 = psd(amp,0,4)
        psd2 = psd(amp,4,8)
        psd3 = psd(amp,8,13)
        psd4 = psd(amp,13,20)
        psd5 = psd(amp,20,30)
        psd6 = psd(amp,30,40)
        psd7 = psd(amp,40,60)
        psd8 = psd(amp,60,80)
        psd9 = psd(amp,80,100)
        psd10 = psd(amp,100,128)
        psds = [psd1, psd2, psd3, psd4, psd5, psd6, psd7, psd8, psd9, psd10]

    else:
        print("Select correct sample rate!")
        exit(1)
    
    return psds

def spectrogram_feature(signal, samplerate, feature_samplerate):
    freq_resolution = 2 # higher the better resolution but time consuming...
    nperseg = 8
    noverlap = 150  #0.75 s
    nfft = int(samplerate * freq_resolution)
    freqs, times, spec = stft(signal, samplerate, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=False, boundary=None)
    amp = (np.log(np.abs(spec) + 1e-10))
    return freqs, times, amp

def convert_to_univariate_eeg(eeg):
    # concatenate all channels into one channel
    uni_eeg = np.concatenate(eeg, axis=1)
    return uni_eeg

def generate_psd_feature(X_train, epoch_duration, samplerate):
    psd_feature_all_epochs = []
    for X_sample in X_train: #X_sample is a 2D array of shape (n_channels, n_samples), that is (19, 1024)
        # read every channel in X_sample.
        psd_feature_multi_channel = []
        for ch in range(X_sample.shape[0]):
            uni_ch = X_sample[ch,:]
            w = hann(epoch_duration*samplerate)
            SFT = ShortTimeFFT(w, hop=epoch_duration*samplerate+1, fs=samplerate, scale_to='psd') #edit mfft to adjust the frequency resolution
            # SFT.k_max = epoch_duration*samplerate        
            Sx = SFT.stft(uni_ch)#(513, 2)
            
            # t_lo, t_hi = SFT.extent(samplerate*epoch_duration)[:2] 
            # print("t_lo: ", t_lo, "t_hi: ", t_hi)
            psd_values = Sx[:, 0]#(513,2)

            if samplerate == 128:
                
                psd1 = psd_values[0:4]
                psd2 = psd_values[4:8]
                psd3 = psd_values[8:13]
                psd4 = psd_values[13:20]
                psd5 = psd_values[20:30]
                psd6 = psd_values[30:40]
                psd7 = psd_values[40:60]
                psd_feature_per_channel = [psd1, psd2, psd3, psd4, psd5, psd6, psd7]
                # psd 1-7 is array with different length, so we need to concatenate them into one array
                psd_feature_per_channel = np.concatenate(psd_feature_per_channel, axis=0)
                #vertically append psd_feature_per_channel to psd_feature_multi_channel
                psd_feature_multi_channel = psd_feature_per_channel if len(psd_feature_multi_channel) == 0 else np.vstack((psd_feature_multi_channel, psd_feature_per_channel))
                print("psd_feature_multi_channel.shape: ", psd_feature_multi_channel.shape)

        psd_feature_all_epochs.append(psd_feature_multi_channel)
    print("psd_feature_all_epochs.shape: ", np.array(psd_feature_all_epochs).shape)
    return np.array(psd_feature_all_epochs)

def main():

    with open('./config_fft.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config['timestamp'] = datetime.now().strftime("%m%d_%H%M%S") # add unique timestamp to the config
        config['dataset'] = config['bids_root'].split('/')[-1]

    os.makedirs(config['preprocessed_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['result_dir'], exist_ok=True)

    # train_epochs, test_epochs = load_dataset(config)
    # load data
    setup_mne_for_processing(verbose=config.get('mne_verbose', False))

    print("\n=== Preprocess Pipeline with Balanced Epochs ===")
    
    # Read and label all events
    all_events, (seizure_subjects, bckg_subjects) = read_all_events(config)
    # train test split, using stratified k-fold cross validation
    split_seed = config.get('split_seed')
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    all_subjects = np.concatenate([seizure_subjects, bckg_subjects])
    all_labels = np.concatenate([np.ones(len(seizure_subjects)), np.zeros(len(bckg_subjects))])
    split_counter = 0
    for train_index, test_index in kf.split(all_subjects, all_labels):
        split_counter += 1
        train_subjects = all_subjects[train_index]
        test_subjects = all_subjects[test_index]
        print(f"Train subjects ids for split {split_counter}: {train_subjects}, train labels: {all_labels[train_index]}")
        print(f"Test subjects ids for split {split_counter}: {test_subjects}, test labels: {all_labels[test_index]}")


    # train events are all events from train subjects
        train_events = all_events[all_events['subject'].isin(train_subjects)]
    # test events are all events from test subjects
        test_events = all_events[all_events['subject'].isin(test_subjects)]

        if config.get('debug', False):
            print("\nReading and labeling events for split {}...".format(split_counter))
            if all_events.empty:
                print("No events found! Check your BIDS directory.")
                break
            print(f"Total events loaded for split {split_counter}: {len(all_events)}")
    
        # Create balanced epochs
        print(f"Creating balanced epochs for train set for split {split_counter}...")
        train_epochs = create_balanced_epochs(train_events, config)
        print(f"Creating balanced epochs for test set for split {split_counter}...")
        test_epochs = create_balanced_epochs(test_events, config)

        X_train, y_train = load_epoch_data(train_epochs, config, split_name='train', split_counter=split_counter, cv=True)
        print("X_train.shape: ", X_train.shape) #(5651, 19, 1024)
        epoch_duration = config.get('epoch_duration')
        samplerate = config.get('sample_rate')
        psd_feature_all_epochs = generate_psd_feature(X_train, epoch_duration, samplerate)
        
        
        # return np.array(psd_feature_all_epochs)

                    




    # psds = power_spectral_density_feature(uni_eeg_epoch, 256)
    # freqs, times, spec = spectrogram_feature(uni_eeg, 256)
    # plt.figure()
    # plt.subplot(11, 1, 1)
    # plt.plot(x_time, uni_eeg)      
    # plt.title('Original Signal (P3-REF)')  
    # plt.subplot(11, 1, 2)
    # plt.plot(psds[0])
    # plt.title('PSD 0~4 Hz feature')  
    # plt.subplot(11, 1, 3)
    # plt.plot(psds[1])
    # plt.title('PSD 4~7 Hz feature')  
    # plt.subplot(11, 1, 4)
    # plt.plot(psds[2])
    # plt.title('PSD 7~13 Hz feature')  
    # plt.subplot(11, 1, 5)
    # plt.plot(psds[3])
    # plt.title('PSD 13~15 Hz feature')  
    # plt.subplot(11, 1, 6)
    # plt.plot(psds[4])
    # plt.title('PSD 15~30 Hz feature')  
    # plt.subplot(11, 1, 7)
    # plt.plot(psds[5])
    # plt.title('PSD 30~45 Hz feature')  
    # plt.subplot(11, 1, 8)
    # plt.plot(psds[6])
    # plt.title('PSD 45~60 Hz feature')  
    # plt.subplot(11, 1, 9)
    # plt.plot(psds[7])
    # plt.title('PSD 60~75 Hz feature')  
    # plt.subplot(11, 1, 10)
    # plt.plot(psds[8])
    # plt.title('PSD 75~97 Hz feature')  
    # plt.subplot(11, 1, 11)
    # plt.plot(psds[9])
    # plt.title('PSD 97~128 Hz feature')  
    # plt.show()

    # plt.figure()
    # plt.pcolormesh(spec)
    # plt.title("spectrogram feature")
    # plt.show()

if __name__ == "__main__":
    main()