from mne import make_fixed_length_events
from mne.io import read_raw_edf

edf_path = "E:\\BIDS_TUSZ\\sub-317\\ses-01\\eeg\\sub-317_ses-01_task-szMonitoring_run-07_eeg.edf"
raw = read_raw_edf(edf_path, preload=False)
print(f"采样率: {raw.info['sfreq']} Hz")
print(f"总时长: {raw.times[-1]} sec")

try:
    events = make_fixed_length_events(raw, duration=10, overlap=0)
    print(f"成功生成 {len(events)} 个事件")
except Exception as e:
    print(f"错误: {e}")