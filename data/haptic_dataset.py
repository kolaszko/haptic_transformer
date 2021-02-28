import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt


class HapticDataset(Dataset):
    def __init__(self, path, key, signal_start=90, signal_length=90):

        with open(path, 'rb') as f:
            pickled = pickle.load(f, encoding='latin1')
            self.signals = pickled[key]

        self.stats = pickled['signal_stats']
        self.weights = pickled['classes_weights']
        self.signal_start = signal_start
        self.signal_length = signal_length


    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        sig = self.signals[index]['signal'][self.signal_start : self.signal_start+self.signal_length, : 3]
        label = self.signals[index]['label']

        return sig, label




