import argparse
import os
import socket
import time
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score
import numpy as np
from utils import summary
from data import HapticDataset
from models import HAPTR, TemporalConvNet
from utils import validate_and_save, save_statistics


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = {}

    torch.manual_seed(42)

    model = torch.load(os.path.join(args.model_path, 'test_model'))
    model.to(device)
    model.train(False)

    summary(model, input_size=(160, 6))

    dummy_input = torch.randn(1, 160, 6, dtype=torch.float).to(device)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    with torch.no_grad():
        for rep in range(repetitions):
            start_time = time.time()
            out = model(dummy_input.unsqueeze(1))
            _, predicted = torch.max(out.data, 1)
            end_time = time.time()
            timings[rep] = (end_time - start_time) * 1000


    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)

    results['mean'] = mean_syn
    results['std'] = std_syn
    results['unit'] = 'ms'

    with open(os.path.join(args.model_path, 'inference_time.json'), 'w') as f:
        f.write(json.dumps(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)

    args, _ = parser.parse_known_args()
    main(args)


