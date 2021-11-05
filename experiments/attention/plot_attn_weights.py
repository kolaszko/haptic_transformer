import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()


def visualize_attention_weights_on_plot(x, y, heads, y_padding=0.2, fig_name="figure"):
    fig, axes = plt.subplots(heads.shape[0], 1, figsize=(5, 5), sharex=True)
    fig.tight_layout()

    y_max, y_min = y.max() + y_padding, y.min() - y_padding

    for head, ax in zip(heads, axes):
        ax.grid(False)

        # add attention weights as a heatmap
        colors = head.reshape(1, -1)
        ax.imshow(colors,
                  extent=[x[0], x[-1], y_min, y_max],
                  aspect='auto',
                  cmap='Blues',
                  interpolation='bilinear')

        # plot each signal on the heatmap
        for s in signal.T:
            sns.lineplot(x=time, y=s, ax=ax, linewidth=2.5)

    plt.savefig("./{}.png".format(fig_name))


# load data
BATCH_IDX = 0
SIGNAL_PATH = "./x.npy"
ATTENTION_WEIGHTS_PATHS = glob.glob("./att_mat_*.npy")
signal = np.load(SIGNAL_PATH, allow_pickle=True)[BATCH_IDX]
time = np.linspace(0, signal.shape[0], signal.shape[0])
attn_weights = list()
for path in ATTENTION_WEIGHTS_PATHS:
    data = np.load(path, allow_pickle=True)[BATCH_IDX].sum(-2)
    attn_weights.append(data)

# draw attention weights
for i, head in enumerate(attn_weights):
    visualize_attention_weights_on_plot(time, signal, head, fig_name="weights_{}".format(i))
