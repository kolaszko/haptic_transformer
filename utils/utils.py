import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score
from torchsummary import summary


def validate_and_save(model, ds_loader, name, device, input_size, base_path):
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

    summary(model, input_size=input_size)

    stat_path = os.path.join(base_path, name)

    if not os.path.exists(stat_path):
        os.makedirs(stat_path)

    with open(f'{stat_path}/acc.npy', 'wb') as f:
        np.save(f, accuracy)

    with open(f'{stat_path}/cm.npy', 'wb') as f:
        np.save(f, cf_matrix)

    with open(f'{stat_path}/miou.npy', 'wb') as f:
        np.save(f, miou)
