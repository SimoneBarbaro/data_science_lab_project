from src.clustering.clustering import SomClusterer
from src.data.read_data import *
from src.dimensionality_reduction.tsne import tsne_dimred
from src.dimensionality_reduction.umap import umap_dimred
from src.visualize.visualize import plot_embedded_cluster
from src.dimensionality_reduction.som import som_embedd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse


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
    labels = clusterer.fit(data).labels_
    embedding = embedding_fn(data, **embedding_args)
    plot_embedded_cluster(embedding, labels)


def run_som_cluster():
    """
    Run with first som and then clustering.
    """
    data = load_sample(frac=0.1, random_state=1, save=False)
    som = SomClusterer(data)
    run_clustering(som, som.get_embeddings)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_embedding', type=str, choices=["none", "tsne", "umap", "som"],
                        help='Choose an embedding to test', default="none")
    parser.add_argument('--test_clustering', type=str, choices=["none", "som_cluster", "kmeans_tsne",
                                                                "kmeans_som", "kmeans_elbow"],
                        help='Choose a clustering run to test', default="none")

    args = parser.parse_args()
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
