# Adapted from https://data.csiro.au/collection/csiro:46885v2

# The dataset has two different sensor measurements from the quadruped robot DyRET:
# 4x force sensors on the feet (referred as raw) and an IMU sensor on the body (imu).
# The first number (0-5) shows the type to the terrain:
# 0. Concrete, 1. Grass, 2. Gravel, 3. Mulch, 4. Dirt, 5. Sand.
# Inside each file there are 10 trials: the forth column (third if starting from 0).
# The second number (1-6) represents the robot speed in each run.
# There were 8 steps for each trial, which means that we have: 6*10*6*8 = 2880 samples.
# The jupyter notebook code for processing the force sensor data and IMU data are provided.
# They give us 2880 force or IMU  samples.

import csv
import pickle

import numpy as np
from torch.utils.data import Dataset

import submodules.haptic_transformer.data.helpers as helpers


class QCATDataset(Dataset):
    def __init__(self, pick_modalities, split_modalities=False, signal_start=90, signal_length=90, standarize=True):

        # placeholders for the data
        self.signals = None
        self.mean = None
        self.std = None
        self.weights = None
        self.num_classes = None

        self.pick_modalities = pick_modalities
        self.dim_modalities = helpers.determine_dim_size([12, 10], pick_modalities)
        self.num_modalities = len(self.pick_modalities)
        self.split_modalities = split_modalities
        self.signal_start = signal_start
        self.signal_length = signal_length
        if standarize:
            self._standarize()

    def _standarize(self):
        if self.signals is not None:
            # do not normalize quaternions
            self.signals['fimu'][..., :-4] = (self.signals['fimu'][..., :-4] - self.mean[..., :-4]) / self.std[..., :-4]

    def __len__(self):
        if self.signals is not None:
            return len(self.signals["fimu"])
        return 0

    def __getitem__(self, index):
        ts = self.signals['fimu'][index, self.signal_start: self.signal_start + self.signal_length]
        sig = helpers.prepare_batch(ts, self.split_modalities, self.pick_modalities, self.dim_modalities)
        label = self.signals['label'][index]
        return sig, label

    def __add__(self, other_qcat):
        self.mean = (self.mean + other_qcat.mean) / 2.0
        self.std = (self.std + other_qcat.std) / 2.0

        self.signals["fimu"] = np.concatenate([self.signals["fimu"], other_qcat.signals["fimu"]], 0)
        self.signals["label_one_hot"] = np.concatenate(
            [self.signals["label_one_hot"], other_qcat.signals["label_one_hot"]], 0)
        self.signals["label"] = np.concatenate([self.signals["label"], other_qcat.signals["label"]], 0)
        return self

    # dedicated classmethods to load QCAT dataset from raw data or from pickle (faster)
    @classmethod
    def from_pickle(cls, path, key, pick_modalities, split_modalities=False, signal_start=90, signal_length=90,
                    standarize=True):

        with open(path, 'rb') as f:
            pickled = pickle.load(f)
            data = pickled[key]  # load QCATDataset instance

        qcat = cls(pick_modalities, split_modalities, signal_start, signal_length, standarize)
        qcat.signals = data.signals
        qcat.mean = data.mean
        qcat.std = data.std
        qcat.weights = data.weights
        qcat.num_classes = data.num_classes
        return qcat

    @classmethod
    def from_data(cls, path, key, pick_modalities, split_modalities=False, signal_start=90, signal_length=90,
                  standarize=True):
        num_classes = 6  # we have 6 terrain classes
        num_steps = 8  # the robot walked 8 steps on each terrain
        max_steps = 662  # this is obtained based on our data
        all_force_colms = 16  # this is based on number of all colms in the csv files
        all_imu_colms = 14  # this is based on number of all colms in the csv files
        relevant_force_colms = 12  # this is our IMU sensor information in the csv files
        relevant_imu_colms = 10  # this is our IMU sensor information in the csv files

        # train
        min_num_speed = 1
        max_num_speed = 6
        min_num_trial = 0
        max_num_trial = 8

        if "val" in key:
            min_num_trial = 8
            max_num_trial = 10

        if "test" in key:
            min_num_speed = 5
            max_num_speed = 6
            min_num_trial = 0
            max_num_trial = 10

        forces, force_labels = read_data(path, num_classes, min_num_trial, max_num_trial, num_steps,
                                         max_steps, all_force_colms, relevant_force_colms, min_num_speed, max_num_speed,
                                         "raw")
        imu, imu_labels = read_data(path, num_classes, min_num_trial, max_num_trial, num_steps, max_steps,
                                    all_imu_colms, relevant_imu_colms, min_num_speed, max_num_speed, "imu")

        signals = {
            "fimu": np.concatenate([forces, imu], -1),
            "label_one_hot": force_labels,
            "label": np.argmax(force_labels, -1)
        }

        # set loaded data slots
        qcat = cls(pick_modalities, split_modalities, signal_start, signal_length, standarize)
        qcat.signals = signals
        qcat.mean = np.mean(signals["fimu"], (0, 1), keepdims=True)
        qcat.std = np.std(signals["fimu"], (0, 1), keepdims=True)
        qcat.weights = [1.0 for _ in range(num_classes)]  # no info about weights of classes
        qcat.num_classes = num_classes
        return qcat


def read_data(folder_path,
              num_classes,
              min_num_trial,
              max_num_trial,
              num_steps,
              max_steps,
              all_colms,
              relevant_colms,
              min_num_speed,
              max_num_speed,
              file_suffix):
    all_seq = num_classes * (max_num_speed - min_num_speed) * (max_num_trial - min_num_trial) * num_steps
    all_data = np.zeros([all_seq, max_steps, all_colms])
    num_trials = max_num_trial - min_num_trial
    data_labels_array = np.zeros((all_seq, num_classes))
    data_length_array = np.zeros((all_seq))
    data_length_array = data_length_array.astype(int)

    cnt = 0
    for i in range(num_classes):
        for j in range(min_num_speed, max_num_speed):  # different speeds
            struct = f'{folder_path}/%d_%d_legSensors_{file_suffix}.csv' % (i, j)
            tmp_data = list(read_lines(struct, min_num_trial, max_num_trial))
            tmp_arr = np.array(tmp_data)
            step, tmp_list = step_count(tmp_arr, num_trials, num_steps)
            step = int(step)
            for k in range(num_trials):
                for l in range(num_steps):
                    all_data[cnt, 0:step, :] = tmp_list[k][l * step:(l + 1) * step]
                    data_labels_array[cnt, i] = 1.0
                    data_length_array[cnt] = step
                    cnt += 1
    return all_data[:, :, 4:4 + relevant_colms], data_labels_array


def read_lines(file, min_num_trial, max_num_trial):
    with open(file, newline="") as data:
        reader = csv.reader(data)
        ind = 0
        for row in reader:
            if (ind > 0) and int(row[3]) in range(min_num_trial, max_num_trial):  # not to include the first row
                yield [float(i) for i in row]
            ind += 1


def step_count(raw_inp, num_trials, num_steps):
    cnt = 0
    inputs = [[] for _ in range(num_trials)]
    for i in range(raw_inp.shape[0]):
        if i > 0:
            if (raw_inp[i, 3] != raw_inp[i - 1, 3]):  # 3 is the column in csv files that shows the num of tiral
                cnt += 1
        inputs[cnt].append(raw_inp[i])
    minimum = 1000000
    for i in range(num_trials):
        if (len(inputs[i]) < minimum):
            minimum = len(inputs[i])
    each_step = np.floor(minimum / num_steps)
    return each_step, inputs
