import time

import torch


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def measure_interence_time(model, repetitions, device):
    timings = list()
    model.warmup(device, 10, 1)

    with torch.no_grad():
        for rep in range(repetitions):
            start_time = time.time()
            _ = model.warmup(device)
            end_time = time.time()
            timings = (end_time - start_time) * 1000
    return timings
