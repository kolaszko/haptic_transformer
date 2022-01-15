import pickle

from torch.utils.data import Dataset

import submodules.haptic_transformer.data.helpers as helpers


class HapticDataset(Dataset):
    def __init__(self, path, key, pick_modalities, split_modalities=False, signal_start=90, signal_length=90,
                 standarize=True):

        with open(path, 'rb') as f:
            pickled = pickle.load(f)
            self.signals = pickled[key]

        self.num_classes = 8
        self.pick_modalities = pick_modalities
        self.num_modalities = len(self.pick_modalities)
        self.dim_modalities = helpers.determine_dim_size([3, 3], pick_modalities)
        self.split_modalities = split_modalities
        self.mean, self.std = pickled['signal_stats']
        self.weights = pickled['classes_weights']
        self.signal_start = signal_start
        self.signal_length = signal_length
        if standarize:
            self._standarize()

    def _standarize(self):
        for s in self.signals:
            s['signal'] = (s['signal'] - self.mean) / self.std

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        ts = self.signals[index]['signal'][self.signal_start: self.signal_start + self.signal_length]
        sig = helpers.prepare_batch(ts, self.split_modalities, self.pick_modalities, self.dim_modalities)
        label = self.signals[index]['label']
        return sig, label

    def __add__(self, other_haptic):
        self.mean = (self.mean + other_haptic.mean) / 2.0
        self.std = (self.std + other_haptic.std) / 2.0
        self.signals += other_haptic.signals
        self.weights = (self.weights + other_haptic.weights) / 2.0
        return self
