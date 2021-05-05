import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup as warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, jaccard_score
from torchsummary import summary
import numpy as np
import colorsys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn import decomposition


from data import HapticDataset
from models import HAPTR

params = {
    'num_classes': 8,
    'projection_dim': 16,
    'hidden_dim': 160,
    'nheads': 16,
    'num_encoder_layers': 1,
    'mlp_head_dim_1': 64,
    'mlp_head_dim_2': 32,
    'dropout': 0.5,
    'lr': 1e-4,
    'gamma': 0.999,
    'weight_decay': 1e-4,
    'comment': 'Add dropout to transformer'
}

def colored_labels(y):
    N = 9
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    distinct_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    y_rgb = list()
    cnt = 0
    for i in y:
        # if i > 0 and y[i - 1] != y[i]:
        #     cnt += 1
        y_rgb.append(distinct_colors[i])

    c = np.asarray(y_rgb) / np.max(y_rgb)
    return c

def pca(data: np.ndarray, labels: np.ndarray):
    X = data.copy()
    y = labels.copy()
    n_components = 3

    fig = plt.figure(1, figsize=(20, 13))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .9, 0.9], elev=50, azim=45)
    plt.cla()

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    x = pca.transform(X)
    c = colored_labels(y)

    _, class_idxs, class_cnts = np.unique(labels, return_index=True, return_counts=True)
    for start, size in zip(class_idxs, class_cnts):
        stop = start + size
        colors = c[start:stop, :]
        ax.scatter(x[start:stop, 0], x[start:stop, 1], x[start:stop, 2], c=colors, cmap='Dark2', s=150, edgecolors='k',
                   label=labels[start])

    ax.legend(loc="best", title="Classes", fontsize="x-large", title_fontsize="x-large")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    ax.grid(True)
    # plt.savefig('./images/pca.png', bbox_inches='tight')
    plt.show()

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    val_ds = HapticDataset(args.dataset_path, 'val_ds', signal_start=0, signal_length=params['hidden_dim'])
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)

    mlp_head_dims = (params['mlp_head_dim_1'], params['mlp_head_dim_2'])

    model = HAPTR(params['num_classes'], params['projection_dim'],
                  params['hidden_dim'], params['nheads'],
                  params['num_encoder_layers'], mlp_head_dims,
                  params['dropout'], next_prediction=True, analysis=True)

    model = torch.load('/home/titan/Projects/mlys/haptic_transformer/runs/Apr30_15-03-46_titan/model')
    model.to(device)
    model.train(False)
    model.analysis = True

    y_pred = []
    y_true = []

    with torch.no_grad():
        for step, data in enumerate(val_dataloader):
            s, l = data[0].to(device), data[1].to(device)

            out = model(s.unsqueeze(1))
            y_pred.extend(out.data.cpu().flatten(1, 2).numpy())

            y_true.extend(l.data.cpu().numpy())


    pca(y_pred, y_true)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1024)

    args, _ = parser.parse_known_args()
    main(args)
