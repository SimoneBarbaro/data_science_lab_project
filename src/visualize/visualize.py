import matplotlib.pyplot as plt
import numpy as np


def visualize_pygal_scatter(embedded):
    """
    Simple SVG scatter plot with Pygal (requires installation).
    :param embedded: A numpy array with two columns.
    """
    import pygal  # pip install pygal
    from IPython.display import SVG, display  # Needed for Jupyter Notebooks, not sure how it works elsewhere
    from pygal.style import CleanStyle
    xy_chart = pygal.XY(stroke=False, style=CleanStyle)
    xy_chart.title = 'TSNE'
    xy_chart.add('TSNE', embedded)
    display(SVG(xy_chart.render(disable_xml_declaration=True)))


def plot_embedded_cluster(embeddings, cluster_labels, save_fig_path=None):
    plt.title('Number of clusters: %d' % len(np.unique(cluster_labels)))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, s=5, cmap='gist_ncar')
    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    plt.show()
