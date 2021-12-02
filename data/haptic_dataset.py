import pickle

from torch.utils.data import Dataset


class HapticDataset(Dataset):
    def __init__(self, path, key, pick_modalities, split_modalities=False, signal_start=90, signal_length=90,
                 standarize=True):

        with open(path, 'rb') as f:
            pickled = pickle.load(f)
            self.signals = pickled[key]

        self.num_classes = 8
        self.pick_modalities = pick_modalities
        self.num_modalities = len(self.pick_modalities)

        self.dim_modalities = [3, 3]
        if len(self.pick_modalities) < len(self.dim_modalities):
            self.dim_modalities = [self.dim_modalities[i] for i in range(len(self.pick_modalities))]

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
        if self.split_modalities:
            sig = list()
            for mod_idx, mod_dim in zip(self.pick_modalities, self.dim_modalities):
                sig.append(self.signals[index]['signal'][self.signal_start: self.signal_start + self.signal_length,
                           mod_idx:mod_idx + mod_dim])

        else:
            sig = [self.signals[index]['signal'][self.signal_start: self.signal_start + self.signal_length]]

        label = self.signals[index]['label']
        return sig, label
