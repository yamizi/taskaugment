from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib

import numpy as np
from utils import get_comb_label


def multipage(filename, figs=None, dpi=200, clear=False):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
        if clear:
            plt.close(fig)
    pp.close()


def plot_tsne(cat,tsne_results, nb_classes=5,classes_to_show=None, split=None, title="", xmin=-80, xmax=80,  ymin=-80, ymax=80):

    colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.XKCD_COLORS)
    fig_size = (25, 25)
    n = 0

    color_map = np.argmax(cat, axis=1)
    if classes_to_show is None:
        classes_to_show = range(nb_classes)

    if split is None:
        fig = plt.figure(figsize=fig_size)
        for  i, cl in enumerate(classes_to_show):
            indices = np.where(color_map == cl)
            indices = indices[0]
            x = tsne_results[indices, 0]
            y = tsne_results[indices, 1]

            color = colors[i] if i < len(colors) else np.random.rand(3, )
            plt.scatter(x, y, label=get_comb_label(cl), c=color)


        for  cl in range(np.max(color_map)):
            if cl in classes_to_show:
                continue
            indices = np.where(color_map == cl)
            indices = indices[0]
            x = tsne_results[indices, 0]
            y = tsne_results[indices, 1]
            color = "grey"
            if len(x):
                plt.scatter(x, y, label="*", c=color, alpha=0.5, marker="*")

        plt.title(title)
        plt.legend(loc='upper right')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        return [fig]
    else:
        fig1 = plt.figure(figsize=fig_size)
        for i, cl in enumerate(classes_to_show):
            indices = np.where(color_map[:split] == cl)
            indices = indices[0]
            if len(indices):
                x1 = tsne_results[indices, 0]
                y1 = tsne_results[indices, 1]

                color = colors[i] if i<len(colors) else np.random.rand(3,)
                plt.scatter(x1, y1, label=get_comb_label(cl), marker=".", alpha=1, c=color)
        plt.title(title)
        plt.legend(loc='upper right')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        fig2 = plt.figure(figsize=fig_size)
        for i, cl in enumerate(classes_to_show):

            indices = np.where(color_map[split:] == cl)
            indices = indices[0]
            if len(indices):
                x2 = tsne_results[indices, 0]
                y2 = tsne_results[indices, 1]
                color = colors[i] if i < len(colors) else np.random.rand(3, )
                plt.scatter(x2, y2, label=get_comb_label(cl), marker=".", alpha=1,c=color)
        plt.title(title)
        plt.legend(loc='upper right')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        return [fig1, fig2]




def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
