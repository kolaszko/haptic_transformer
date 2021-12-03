import io

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def fig2numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


def impose(weights, signal, colormaps):
    assert len(signal.shape) == 3
    assert len(weights.shape) == 2

    # create a figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.tight_layout()
    ax.grid(False)

    # x - axis
    s_len = signal.shape[0]
    t = np.linspace(0, s_len, s_len)

    # y - axis
    # y_max, y_min = signal.max() + 0.1, signal.min() - 0.1
    y_max, y_min = -3.5, 3.5
    norm_w = (weights - weights.min(0)) / (weights.max(0) - weights.min(0))
    leading = weights.argmax(-1)
    w = (leading + norm_w.max(-1))[np.newaxis]

    # concatenate colormaps of modalities
    modalities = np.split(signal, signal.shape[-1], -1)
    n_mod = len(modalities)
    mycolormap = mcolors.LinearSegmentedColormap.from_list('mycolormap', np.concatenate(colormaps, 0))

    # show a background from attention weights
    ax.imshow(w,
              alpha=0.6,
              extent=[t[0], t[-1], y_min, y_max],
              aspect='auto',
              cmap=mycolormap,
              interpolation='bilinear')

    # impose modalities on the background in corresponding colormaps
    for i in range(n_mod):
        y = modalities[i]  # modality signals
        c = colormaps[i].mean(0)  # mean color for the corresponding colormap
        ax.plot(t, y.squeeze(-1), linewidth=1.5, c=c)

    # save a figure to the numpy array
    img = fig2numpy(fig)
    plt.close(fig)
    return img


def create_image(img, w=400, h=200, dtype=np.uint8):
    img = resize(img, (h, w), preserve_range=True)
    img = (img - img.min()) / (img.max() - img.min())
    return np.multiply(img, 256).astype(dtype)
