import json
import os
import socket
from datetime import datetime

import imageio
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from torchsummary import summary


class GIF:
    def __init__(self, folder, filename="result.gif", w=400, h=300):
        os.makedirs(folder, exist_ok=True)
        self._folder = folder
        self._gif_path = os.path.join(folder, filename)
        self._writer = imageio.get_writer(self._gif_path, mode='I')
        self._w = w
        self._h = h
        self._filenames = []
        self._next = 0

    def add(self, img):
        if len(img.shape) == 2 or len(img.shape) == 3:
            filename = os.path.join(self._folder, f'{self._next}.png')
            imageio.imwrite(filename, img)
            self._filenames.append(filename)
            self._next += 1

    def save(self, remove_files=False):
        for filename in self._filenames:
            image = imageio.imread(filename)
            self._writer.append_data(image)

            if remove_files:
                os.remove(filename)

        if remove_files:
            self._filenames.clear()

        print(f'gif saved at {self._gif_path}')


def save_json(data, filepath):
    with open(filepath, 'w') as f:
        f.write(json.dumps(data))


def save_numpy(data, filepath):
    with open(filepath, 'wb') as f:
        np.save(f, data)


def save_dict(d, filepath):
    with open(filepath, 'w') as f:
        print(d, file=f)


def save_statistics(y_true, y_pred, model, path, input_size):
    cf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    miou = jaccard_score(y_true, y_pred, average='weighted')
    trainable_params = summary(model, input_size=input_size)
    stat_path = os.path.join(path, 'statistics')

    if not os.path.exists(stat_path):
        os.makedirs(stat_path)

    stats = {
        'acc': accuracy,
        'miou': miou,
        'params': trainable_params.__repr__(),
    }

    save_json(stats, f'{stat_path}/stats.json')
    save_numpy(accuracy, f'{stat_path}/acc.npy')
    save_numpy(cf_matrix, f'{stat_path}/cm.npy')
    save_numpy(miou, f'{stat_path}/miou.npy')

    with open(f'{stat_path}/stats.txt', 'w') as f:
        f.write(str(trainable_params.__repr__()))


def logdir_name(folder, experiment_name):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(folder, experiment_name, current_time + '_' + socket.gethostname())
    os.makedirs(log_dir, exist_ok=True)
    return log_dir
