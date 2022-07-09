import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects


# Random state.
RS = 20180101

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def scatter(x, colors, mode):
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    if mode == 'raw':
        # 原始数据可视化图
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40)
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')
        return f, ax, sc
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(7):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i + 1), fontsize=34)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txt


def pic(feature, label, name, mode):

    X = feature
    y = label

    digits_proj = TSNE(random_state=RS).fit_transform(X)
    scatter(digits_proj, y.numpy(), mode)
    foo_fig = plt.gcf()  # 'get current figure'
    foo_fig.savefig(name, format='png', dpi=1000)
    # plt.show()
