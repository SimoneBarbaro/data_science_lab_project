"""
This file is for functions that search the best configuration of clustering parameters, like finding the k on kmeans.
Only methods that require human supervision should be here,
automatic methods should be incorporated into the fit method of a clusterer instead.
"""
import argparse
import json
import os

from sklearn.metrics import make_scorer, silhouette_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from clustering.clustering import Clusterer, get_clusterer
from data.read_data import load_sample, load_sample_with_names, filter_twosides, get_twosides_meddra
from dimensionality_reduction.embedding import get_embedder


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


class GenericClusterer(Clusterer):
    def __init__(self, clusterer, **kwargs):
        self.clusterer = get_clusterer(clusterer, **kwargs)

    def fit_impl(self, data):
        self.clusterer.fit_impl(data)

    def predict(self, data):
        return self.clusterer.predict(data)


class ParamSearch:
    def __init__(self, search_config, metrics=silhouette_score):
        self.search_config = search_config
        self.scorer = make_scorer(metrics)

    def search(self, data, min_coverage=30):
        X = StandardScaler().fit_transform(data)
        search = RandomizedSearchCV(GenericClusterer, param_distributions=self.search_config,
                                    n_iter=min_coverage, scoring=self.scorer, cv=1, refit=False)
        search.fit(X)
        return pd.DataFrame(search.cv_results_)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--clustering', type=str, choices=["som_cluster", "kmeans", "dpgmm", "gmm", "dbscan", "aggl"],
                        help='Choose a clustering method', default="kmeans")
    parser.add_argument('--clustering_search_config', type=str,
                        help='Choose a clustering configuration file for the search',
                        default="../config/kmeans_search_test.json")

    parser.add_argument('--embedding', type=str, choices=["tsne", "umap", "som"],
                        help='Choose an embedding method', default="tsne")
    parser.add_argument('--embedding_config', type=str,
                        help='Choose a embedding configuration file', default="../config/tsne_default.json")

    parser.add_argument('--normalize', action='store_true', default=False,
                        help="add if you want to do normalize the columns of the target prediction dataset")
    parser.add_argument('--pre_embed', action='store_true', default=False,
                        help="add if you want to do the embedding before clustering")
    parser.add_argument('--pre_filter', action='store_true', default=False,
                        help="add to filter results before clustering based on twosides")

    parser.add_argument('--random_seed', type=int, default=42, help="global seed for random functions")
    parser.add_argument('--test', action='store_true', default=False,
                        help="add if you want to test the run on a smaller amount of data")
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    frac = 0.1 if args.test else 1

    data, names = load_sample_with_names(frac=frac, random_state=args.random_seed)
    if args.normalize:
        data = StandardScaler().fit_transform(data)
    print(os.listdir("./"))
    with open(args.clustering_search_config) as f:
        clustering_search_config = json.load(f)
    with open(args.embedding_config) as f:
        embedding_args = json.load(f)

    embedder = get_embedder(args.embedding, random_state=args.random_seed, **embedding_args)

    if args.pre_filter:
        data, names = filter_twosides(data, names, get_twosides_meddra(False))

    if args.pre_embedd:
        data = embedder.embed(data)

    ParamSearch(clustering_search_config).search(data)
