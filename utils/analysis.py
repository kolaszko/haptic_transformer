import io

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

CMAPS = ['Reds', 'Blues', 'Greens']


def fig2numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


def impose(weights, signal):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.tight_layout()
    ax.grid(False)

    signal_length = signal.shape[0]
    time = np.linspace(0, signal_length, signal_length)

    y_max, y_min = signal.max() + 0.1, signal.min() - 0.1
    norm_w = (weights - weights.min(0)) / (weights.max(0) - weights.min(0))
    leading = weights.argmax(-1)
    w = (leading + norm_w.max(-1))[np.newaxis]

    modalities = np.split(signal, signal.shape[-1], -1)
    colormaps = [plt.cm.get_cmap(name)(np.linspace(0, 1., int(256 / len(modalities)))) for name in CMAPS][
                :len(modalities)]
    colors = np.concatenate(colormaps, 0)
    mycolormap = mcolors.LinearSegmentedColormap.from_list('mycolormap', colors)

    ax.imshow(w,
              alpha=0.6,
              extent=[time[0], time[-1], y_min, y_max],
              aspect='auto',
              cmap=mycolormap,
              interpolation='bilinear')

    for modality, v in zip(modalities, np.linspace(0, leading.max(), leading.max() + 1)):
        ax.plot(time, modality.squeeze(-1), linewidth=1.5, c=mycolormap(v - 0.5))

    img = fig2numpy(fig)
    plt.close(fig)
    return img


def create_image(img, w=400, h=200, dtype=np.uint8):
    img = resize(img, (h, w), preserve_range=True)
    img = (img - img.min()) / (img.max() - img.min())
    return np.multiply(img, 256).astype(dtype)
