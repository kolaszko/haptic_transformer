import torch
from torch.utils.data import Dataset
import pickle


class HapticDataset(Dataset):
    def __init__(self, path, key, signal_start=90, signal_length=90, standarize=True, prediction=False):

        with open(path, 'rb') as f:
            pickled = pickle.load(f, encoding='latin1')
            self.signals = pickled[key]

        self.mean, self.std = pickled['signal_stats']
        self.weights = pickled['classes_weights']
        self.signal_start = signal_start
        self.signal_length = signal_length
        self.prediction = prediction
        if standarize:
            self._standarize()

    def _standarize(self):
        for s in self.signals:
            s['signal'] = (s['signal'] - self.mean) / self.std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        sig = self.signals[index]['signal'][self.signal_start: self.signal_start + self.signal_length]
        label = self.signals[index]['label']

        if self.prediction:
            next_sig = self.signals[index]['signal'][self.signal_start + self.signal_length]
            return sig, label, next_sig

        return sig, label
