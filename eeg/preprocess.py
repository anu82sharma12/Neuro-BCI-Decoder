import mne
import numpy as np

def preprocess_raw(raw):
    raw.filter(8, 30, fir_design='firwin')
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, baseline=None, preload=True)
    data = epochs.get_data()[:, :8, :]  # 8 channels
    labels = epochs.events[:, -1] - 1   # 0-3
    return data, labels
