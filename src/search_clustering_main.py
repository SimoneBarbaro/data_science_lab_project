import argparse
import json
import os

from sklearn.preprocessing import StandardScaler
import numpy as np

from clustering.search_clustering import ParamSearch
from data.read_data import load_sample_with_names, filter_twosides, get_twosides_meddra
from dimensionality_reduction.embedding import get_embedder


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

    parser.add_argument('--normalize', action='store_true', default=True,
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

    with open(args.clustering_search_config) as f:
        clustering_search_config = json.load(f)
    with open(args.embedding_config) as f:
        embedding_args = json.load(f)

    embedder = get_embedder(args.embedding, random_state=args.random_seed, **embedding_args)

    if args.pre_filter:
        data, names = filter_twosides(data, names, get_twosides_meddra(False))

    if args.pre_embed:
        data = embedder.embed(data)

    search_result = ParamSearch(clustering_search_config).search(data)
    print(search_result.sort_values("rank_test_score"))
