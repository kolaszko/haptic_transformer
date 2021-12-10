import argparse

import numpy as np
import torch

from experiments.transformer.transformer_train import accuracy

torch.manual_seed(42)
import matplotlib.pyplot as plt

colormaps = [
    plt.cm.get_cmap('Reds')(np.linspace(0, 1., int(256 / 2))),
    plt.cm.get_cmap('Blues')(np.linspace(0, 1., int(256 / 2)))
]

neutral_cmap = plt.cm.get_cmap('Greens')(np.linspace(0, 1., 256))


def plot_relative_weights_change(weights, noise_amplitudes):
    x = get_range(noise_amplitudes)
    y = np.median(weights.mean(0), 1)

    # relative change of weights for whole signal
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.tight_layout()
    ax.grid(True)
    ax.plot(x, y[:, 0], linewidth=1.5, c=colormaps[0].mean(0))
    ax.plot(x, y[:, 1], linewidth=1.5, c=colormaps[1].mean(0))
    return ax


def barplot_accuracy_to_noise(hits, noise_amplitudes):
    x = get_range(noise_amplitudes)
    y = accuracy(hits, noise_amplitudes.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.tight_layout()
    ax.grid(True)
    ax.set_ylim([0.0, 100.0])
    ax.bar(x, y, width=1. / x.shape[0], color=neutral_cmap.mean(0))
    return ax


def reshape(x):
    return np.concatenate([np.stack(w, 0) for w in x], 1).transpose((1, 0, 2, 3))


def get_range(signal):
    return abs(signal.max(-1).mean((0, -1)) - signal.min(-1).mean((0, -1)))


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def main(args):
    data = np.load(args.input_file, allow_pickle=True).item()

    for noised_mod_idx in data.keys():
        print(f"Noised modality: {noised_mod_idx}")
        weights = reshape(data[noised_mod_idx]['weights'])
        noises = reshape(data[noised_mod_idx]['noises'])
        hits = np.stack(data[noised_mod_idx]['hits'], 0).sum(0)
        ax = plot_relative_weights_change(weights, noises)
        # ax = barplot_accuracy_to_noise(hits, noises)

    # plt.show()
    multipage(f"{args.suffix}.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str,
                        default="/home/mbed/Projects/haptic_transformer/experiments/transformer/PUT_haptrpp_weights_analysis.npy")
    parser.add_argument('--suffix', type=str, default="PUT_haptrpp2")
    args, _ = parser.parse_known_args()
    main(args)
