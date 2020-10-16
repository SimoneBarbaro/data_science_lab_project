import numpy as np

from src.dimensionality_reduction.som import Som


class Clusterer:
    """
    Abstract class to follow the same usage of sklearn clustering classes for our new clustering methods.
    """
    labels_ = None

    def fit_impl(self, data):
        raise NotImplementedError

    def fit(self, data):
        self.fit_impl(data)
        return self


class SomClusterer(Som, Clusterer):
    """
    Clusterer that works on
    """
    def fit(self, data):
        self.train()

    def get_som_clusters(self, data, plot=False):
        clusters = self.net.cluster(data, show=plot, printout=False, savefile=False)
        self.labels_ = self.get_cluster_labels(clusters)
        return clusters

    def get_cluster_labels(self, som_clusters):
        n = self.train_data.shape[0]
        labels = np.zeros(n)
        for c, clust in enumerate(som_clusters):
            for i in clust:
                labels[i] = c
        return labels
