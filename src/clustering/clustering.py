import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, FeatureAgglomeration, OPTICS
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from dimensionality_reduction.som import Som


class Clusterer(BaseEstimator):
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

    def get_params(self, deep=True):
        raise NotImplementedError

    def set_params(self, **params):
        raise NotImplementedError


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


class SklearnPredictClusterer(Clusterer):
    """
    Wrapper for SklearnClusterers with no predict method
    """

    def __init__(self, original_clusterer):
        self.clusterer = original_clusterer

    def get_params(self, deep=True):
        return self.clusterer.get_params(deep)

    def set_params(self, **params):
        return self.clusterer.set_params(**params)

    def fit_impl(self, data):
        self.clusterer.fit(data)

    def predict(self, data):
        return self.clusterer.fit_predict(data)


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
        return SklearnPredictClusterer(DBSCAN(**kwargs))
    elif name == "optics":
        return SklearnPredictClusterer(OPTICS(**kwargs))
    elif name == "mean_shift":
        return SklearnPredictClusterer(MeanShift(**kwargs))
    elif name == "aggl":
        return SklearnPredictClusterer(AgglomerativeClustering(**kwargs))
    elif name == "aggl_features":
        return SklearnPredictClusterer(FeatureAgglomeration(**kwargs))
    else:
        raise NotImplementedError("Clusterer requested not implemented")
