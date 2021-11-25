import json
import os

import imageio
import numpy as np
from skimage.transform import resize
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from torchsummary import summary


def save_gif(img_list, folder, epoch, file_suffix="train", w=300, h=400):
    if len(img_list) > 1:
        path = os.path.join(folder, f"epoch_{file_suffix}_{epoch:05d}.gif")
        with imageio.get_writer(path, mode='I') as writer:
            for img in img_list:
                img = resize(img, (w, h))
                img = (img - img.min()) / (img.max() - img.min())
                img = np.multiply(img, 256).astype(np.uint8)
                writer.append_data(img)
        print(f"Saved GIF at {path}.")
    else:
        print("Empty img list.")


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
