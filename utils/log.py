import json
import os

import imageio
import numpy as np
import torch
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
        'params': str(trainable_params.cpu().numpy()),
    }

    with open(f'{stat_path}/stats.json', 'w') as f:
        f.write(json.dumps(stats))

    with open(f'{stat_path}/acc.npy', 'wb') as f:
        np.save(f, accuracy)

    with open(f'{stat_path}/cm.npy', 'wb') as f:
        np.save(f, cf_matrix)

    with open(f'{stat_path}/miou.npy', 'wb') as f:
        np.save(f, miou)

    with open(f'{stat_path}/stats.txt', 'w') as f:
        f.write(str(trainable_params.cpu().numpy()))


def validate_and_save(model, ds_loader, name, device, input_size):
    model.train(False)
    y_pred = []
    y_true = []

    with torch.no_grad():
        for step, data in enumerate(ds_loader):
            s, labels = data[0].to(device), data[1].to(device)

            out = model(s.unsqueeze(1))
            _, predicted = torch.max(out.data, 1)
            y_pred.extend(predicted.data.cpu().numpy())

            y_true.extend(labels.data.cpu().numpy())

    cf_matrix = confusion_matrix(y_true, y_pred)

    print(cf_matrix)
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy_score(y_true, y_pred))
    miou = jaccard_score(y_true, y_pred, average='weighted')
    print(jaccard_score(y_true, y_pred, average='weighted'))

    trainable_params = summary(model, input_size=input_size)

    print(name)
    stat_path = os.path.join(name, 'statistics')

    print(stat_path)

    if not os.path.exists(stat_path):
        os.makedirs(stat_path)

    stats = {
        'acc': accuracy,
        'miou': miou,
        'params': str(trainable_params.cpu().numpy()),
    }

    with open(f'{stat_path}/stats.json', 'w') as f:
        f.write(json.dumps(stats))

    with open(f'{stat_path}/acc.npy', 'wb') as f:
        np.save(f, accuracy)

    with open(f'{stat_path}/cm.npy', 'wb') as f:
        np.save(f, cf_matrix)

    with open(f'{stat_path}/miou.npy', 'wb') as f:
        np.save(f, miou)

    with open(f'{stat_path}/stats.txt', 'w') as f:
        f.write(str(trainable_params.cpu().numpy()))
