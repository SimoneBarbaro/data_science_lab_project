from src.clustering.clustering import SomClusterer, get_clusterer
from src.data.read_data import *
from src.dimensionality_reduction.embedding import get_embedder
from src.dimensionality_reduction.tsne import tsne_dimred
from src.dimensionality_reduction.umap import umap_dimred
from src.experiment import Experiment
from src.visualize.visualize import plot_embedded_cluster
from src.dimensionality_reduction.som import som_embedd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

import argparse
import json


def run_test_embedding(embedding_fn, **embedding_args):
    data = load_sample(frac=0.1, random_state=1, save=False)
    embedding = embedding_fn(data, **embedding_args)
    plt.plot(embedding[:, 0], embedding[:, 1], '.')
    plt.show()
    # Visualization with Pygal (see visualize.py)
    #visualize_pygal_scatter(embedding)


def run_tsne():
    """
    Run tSNE with a sample of 10% of the original single-drug data.
    """
    run_test_embedding(tsne_dimred, perplexity=40, n_jobs=4, random_state=3)


def run_umap():
    """
    Run UMAP with a sample of 10% of the original single-drug data.
    """
    run_test_embedding(umap_dimred, n_neighbors=100, min_dist=0.0, random_state=2)


def run_som():
    """
    Run UMAP with a sample of 10% of the original single-drug data.
    """
    run_test_embedding(som_embedd)


def run_clustering(clusterer, embedding_fn, **embedding_args):
    data = load_sample(frac=0.1, random_state=1, save=False)
    labels = clusterer.fit(data).predict(data)
    embedding = embedding_fn(data, **embedding_args)
    plot_embedded_cluster(embedding, labels)


def run_som_cluster():
    """
    Run with first som and then clustering.
    """
    data = load_sample(frac=0.1, random_state=1, save=False)
    som = SomClusterer(data)
    run_clustering(som, som.embed)


def run_kmeans_tsne():
    """
    Run k-means clustering with a sample of 10% of the original data,
    and visualize with tSNE.
    """
    k = 10
    run_clustering(KMeans(n_clusters=k, random_state=4), tsne_dimred, perplexity=40, n_jobs=4, random_state=3)


def run_kmeans_som():
    """
    Run k-means clustering with a sample of 10% of the original data,
    and visualize with SOM.
    """
    k = 10
    run_clustering(KMeans(n_clusters=k, random_state=4), som_embedd)


def run_dpgmm_tsne():
    """
    Runs a infinite gaussian mixture model with 5% of data
    """
    run_clustering(BayesianGaussianMixture(n_components=5, covariance_type='full'),
                   tsne_dimred, perplexity=40, n_jobs=4, random_state=3)


def run_gmm_tsne(k=8):
    """
    Runs a gaussian mixture model with 5% of data (this is what worked on local machine)
    """
    run_clustering(GaussianMixture(n_components=k,covariance_type='full'),
                   tsne_dimred, perplexity=40, n_jobs=4, random_state=3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    """
    parser.add_argument('--test_embedding', type=str, choices=["none", "tsne", "umap", "som"],
                        help='Choose an embedding to test', default="none")
    parser.add_argument('--test_clustering', type=str, choices=["none", "som_cluster", "kmeans_tsne",
                                                                "kmeans_som", "kmeans_elbow", "dpgmm_tsne", "gmm_tsne"],
                        help='Choose a clustering run to test', default="none")
    """

    parser.add_argument('--clustering', type=str, choices=["som_cluster", "kmeans", "dpgmm", "gmm"],
                        help='Choose a clustering method', default="kmeans")
    parser.add_argument('--clustering_config', type=str,
                        help='Choose a clustering configuration file', default="../config/kmeans_default.json")
    parser.add_argument('--embedding', type=str, choices=["tsne", "umap", "som"],
                        help='Choose an embedding method', default="tsne")
    parser.add_argument('--embedding_config', type=str,
                        help='Choose a embedding configuration file', default="../config/kmeans_default.json")
    parser.add_argument('--random_seed', type=int, default=42, help="global seed for random functions")
    parser.add_argument('--test', help="add if you want to test the run on a smaller amount of data")
    parser.add_argument('--pre_embed', help="add if you want to do the embedding before clustering")
    parser.add_argument('--pre_filter', help="add to filter results before clustering based on twosides")
    parser.add_argument('run_name', type=str, default="test",
                        help="name of the run, a folder with that name will be created in results to store all the "
                             "relevant results of the run")

    args = parser.parse_args()

    frac = 0.1 if args.test else 1
    print(frac)
    data, names = load_sample_with_names(frac=frac, random_state=args.random_seed)

    clustering_args = json.load(args.clustering_config)
    embedding_args = json.load(args.embedding_config)

    if args.clustering == "som_cluster":
        clustering_args["train_data"] = data

    clusterer = get_clusterer(args.clustering, random_state=args.random_seed, **clustering_args)
    embedder = get_embedder(args.embedding, random_state=args.random_seed, **embedding_args)

    experiment = Experiment(data, names, clusterer, embedder, pre_embedd=args.pre_embedd,
                            pre_filter=args.pre_filter, run_name=args.run_name)

    experiment.run()
    """
    if args.test_embedding == "tsne":
        run_tsne()
    elif args.test_embedding == "umap":
        run_umap()
    elif args.test_embedding == "som":
        run_som()

    if args.test_clustering == "som_cluster":
        run_som_cluster()
    elif args.test_clustering == "kmeans_tsne":
        run_kmeans_tsne()
    elif args.test_clustering == "kmeans_som":
        run_kmeans_som()
    elif args.test_clustering == "kmeans_elbow":
        run_kmeans_elbow()
    elif args.test_clustering == "dpgmm_tsne":
        run_dpgmm_tsne()
    elif args.test_clustering == "gmm_tsne":
        run_gmm_tsne()
    """