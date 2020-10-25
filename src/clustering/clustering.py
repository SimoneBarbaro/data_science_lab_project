import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

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
        # self.train() SOM is trained on creation
        return self

    def predict(self, data):
        self.get_som_clusters(data)
        return self.labels_

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


def get_clusterer(name, **kwargs):
    if name == "som_cluster":
        return SomClusterer(**kwargs)
    elif name == "kmeans":
        return KMeans(**kwargs)
    elif name == "gmm":
        return GaussianMixture(**kwargs)
    elif name == "dpgmm":
        return BayesianGaussianMixture(**kwargs)
    else:
        raise NotImplementedError("Clusterer requested not implemented")
