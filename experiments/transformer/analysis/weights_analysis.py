import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from experiments.transformer.transformer_train import accuracy

torch.manual_seed(42)

CMAPS = [
    plt.cm.get_cmap('Reds')(np.linspace(0, 1., int(256 / 2))),
    plt.cm.get_cmap('Blues')(np.linspace(0, 1., int(256 / 2)))
]

NEUTRAL_CMAP = plt.cm.get_cmap('Greens')(np.linspace(0, 1., 256))

FILES_TO_COMPARE = [
    "/home/mbed/Projects/haptic_transformer/experiments/transformer/analysis/PUT_HAPTR2.npy",
    "/home/mbed/Projects/haptic_transformer/experiments/transformer/analysis/PUT_HAPTR_modatt2.npy"
]

BARPLOT_TITLES = [
    "Noising force modality",
    "Noising torque modality"
]

BARPLOT_LABELS = [
    "HAPTR",
    "HAPTR ModAtt"
]


def plot_relative_weights_change(weights, noise_amplitudes, title):
    x = get_range(noise_amplitudes)
    y = np.mean(np.median(weights, 0), 1)
    y_std = np.mean(np.std(weights, 0), 1)
    y_rel = ((y * 100.0) / y[0]) - 100.0

    # relative change of weights for whole signal
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.grid(True)
    ax.set_ylim([-10.0, 10.0])

    ax.errorbar(x, y_rel[:, 0], y_std[:, 0], linewidth=1.5, c=CMAPS[0].mean(0), label="Force weights")
    ax.errorbar(x, y_rel[:, 1], y_std[:, 1], linewidth=1.5, c=CMAPS[1].mean(0), label="Torque weights")

    ax.legend()
    ax.set_title(title)
    ax.set_ylabel('Relative mean weight change [%]')
    # ax.set_xlabel('Uniform additive noise mean [-]')
    ax.set_xlabel('Uniform additive noise range [-]')
    return ax


def barplot_accuracy_to_noise(hits, noise_amplitudes, color, ax, label, title):
    x = get_range(noise_amplitudes)
    y = accuracy(hits, noise_amplitudes.shape[0])
    ax.grid(True)
    ax.set_ylim([0.0, 100.0])
    ax.bar(x, y, width=1. / x.shape[0], color=color, label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel('Test dataset accuracy [%]')
    # ax.set_xlabel('Uniform additive noise mean [-]')
    ax.set_xlabel('Uniform additive noise range [-]')
    return ax


def plot_accuracy_to_noise(hits, noise_amplitudes, color, ax, label, title):
    x = get_range(noise_amplitudes)
    y = accuracy(hits, noise_amplitudes.shape[0])
    ax.grid(True)
    ax.set_ylim([0.0, 100.0])
    ax.plot(x, y, color=color, label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel('Test dataset accuracy [%]')
    # ax.set_xlabel('Uniform additive noise mean [-]')
    ax.set_xlabel('Uniform additive noise range [-]')
    return ax


def reshape(x):
    return np.concatenate([np.stack(w, 0) for w in x], 1).transpose((1, 0, 2, 3))


def get_range(signal):
    return abs(signal.max(-1).mean((0, -1)) - signal.min(-1).mean((0, -1)))


def get_mean(signal):
    return signal.mean((-1, -2, 0))


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


if __name__ == '__main__':
    fig, axes = plt.subplots(1, len(FILES_TO_COMPARE), figsize=(10, 5))
    # fig.tight_layout()

    for k, path in enumerate(FILES_TO_COMPARE):
        data = np.load(path, allow_pickle=True).item()
        print(f"Processing {path}")

        for i, noised_mod_key in enumerate(data.keys()):
            print(f"Noised modality: {noised_mod_key}")
            noises = reshape(data[noised_mod_key]['noises'])
            hits = np.stack(data[noised_mod_key]['hits'], 0).sum(0)
            color = CMAPS[i].mean(0) - (k / len(FILES_TO_COMPARE)) * CMAPS[i].mean(0)
            # barplot_accuracy_to_noise(hits, noises, color, axes[i], label=BARPLOT_LABELS[k], title=BARPLOT_TITLES[i])
            plot_accuracy_to_noise(hits, noises, color, axes[i], label=BARPLOT_LABELS[k], title=BARPLOT_TITLES[i])

            # if weights available add them on a separate plot
            w = data[noised_mod_key]['weights']
            if type(w) is list and len(w) > 0 and type(w[0]) is list and len(w[0]) > 0:
                weights = reshape(w)
                plot_relative_weights_change(weights, noises, title=BARPLOT_TITLES[i])

    multipage(f"noise_robustness.pdf")
