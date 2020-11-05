"""
This file is for functions that search the best configuration of clustering parameters, like finding the k on kmeans.
Only methods that require human supervision should be here,
automatic methods should be incorporated into the fit method of a clusterer instead.
"""
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd

from data.read_data import load_sample


def run_kmeans_elbow():
    """
    Run k-means clustering with a sample of 10% of the original data,
    and plot the "elbow plot" for k=1,...,29.
    """
    # Read and sample the data, create the matrix (see read_data.py)
    data = load_sample(frac=0.1, random_state=1, save=False)

    # Elbow plot
    from sklearn.cluster import KMeans
    sum_of_squared_distances = []
    K = range(1,30)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        sum_of_squared_distances.append(km.inertia_)

    import matplotlib.pyplot as plt
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum of squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def get_unsupervised_scorer(metric):
    """
    Get an unsupervised scorer for clustering compatible with sklearn parameter search methods.
    :param metric: an unsupervised metric, it must take as input the dataset and the cluster labels
    :return: a scorer based on this metric
    """
    def unsupervised_scorer(estimator, X):
        cluster_labels = estimator.fit_predict(X)  # TODO check if we have it everywhere
        return metric(X, cluster_labels)
    return unsupervised_scorer


def get_scorer_from_metrics(name):
    """
    Return a scorer from a give metrics name.
    :param name: the metrics name.
    :return: a scorer
    """
    if name == "silhouette":
        return get_unsupervised_scorer(silhouette_score)
    else:
        raise NotImplementedError()


class ParamSearch:
    """
    Class for parameter search
    """
    def __init__(self, clusterer, search_config, metric):
        """
        Create a parameter searcher
        :param clusterer: a clusterer
        :param search_config: a dictionary with the parameters as keys
        and as values an array of values representing the options to search from.
        :param metric: the name of an unsupervised clustering metrics
        """
        self.clusterer = clusterer
        self.search_config = search_config
        self.scorer = get_scorer_from_metrics(metric)
        self.full_coverage = 1
        for v in search_config.values():
            if isinstance(v, list):
                self.full_coverage *= len(v)

    def search(self, data, min_coverage=30):
        """
        Start a search on a give dataset
        :param data: the data matrix
        :param min_coverage: the minimum number of parameters to test,
        if the search space is larger, a random search is used with that many iterations, o
        therwise a grid search is used.
        :return: A dataframe with the results in the typical sklearn format
        """
        # Search is not really CV, only one fold is used because the metrics is unsupervised
        fake_cv = [(slice(None), slice(None))]
        if self.full_coverage < min_coverage:
            search = GridSearchCV(self.clusterer, param_grid=self.search_config, scoring=self.scorer,
                                  cv=fake_cv, refit=False)
        else:
            search = RandomizedSearchCV(self.clusterer, param_distributions=self.search_config,
                                        n_iter=min_coverage, scoring=self.scorer,
                                        cv=fake_cv, refit=False)
        search.fit(data)
        return pd.DataFrame(search.cv_results_)
