import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from dimensionality_reduction.som import Som


class Clusterer:
    """
    Abstract class to follow the same usage of sklearn clustering classes for our new clustering methods.
    """
    labels_ = None

    def fit_impl(self, data):
        """
        Implementation of the fit function.
        :param data: data
        """
        raise NotImplementedError

    def fit(self, data):
        """
        Wrapper to the fit function because of compatibility with sklearn fluent interface.
        :param data: data
        :return: self
        """
        self.fit_impl(data)
        return self


class SomClusterer(Som, Clusterer):
    """
    Clusterer that works on Som.
    """
    def fit_impl(self, data):
        # self.train() SOM is trained on creation
        pass

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


class AgglomerativeClusterer(AgglomerativeClustering, Clusterer):
    """
    Wrapper for AgglomerativeClustering because for some strange reason they forgot to put a predict method in it!
    """
    def fit_impl(self, data):
        super(AgglomerativeClusterer, self).fit(data)

    def predict(self, data):
        return self.fit_predict(data)


def get_clusterer(name, **kwargs):
    """
    Return an clusterer given the name.
    :param name: The clusterer name.
    :param kwargs: Arguments to be passed to the clusterer.
    :return: A clusterer.
    """
    if name == "som_cluster":
        return SomClusterer(**kwargs)
    elif name == "kmeans":
        return KMeans(**kwargs)
    elif name == "gmm":
        return GaussianMixture(**kwargs)
    elif name == "dpgmm":
        return BayesianGaussianMixture(**kwargs)
    elif name == "dbscan":
        return DBSCAN(**kwargs)
    elif name == "aggl":
        return AgglomerativeClusterer(**kwargs)
    else:
        raise NotImplementedError("Clusterer requested not implemented")
