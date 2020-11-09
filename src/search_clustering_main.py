import argparse
import json

from sklearn.preprocessing import StandardScaler
import numpy as np

from clustering.clustering import get_clusterer
from clustering.search_clustering import ParamSearch
from data.read_data import load_sample_with_names, filter_twosides, get_twosides_meddra
from dimensionality_reduction.embedding import get_embedder


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--clustering', type=str, choices=["som_cluster", "kmeans", "dpgmm", "gmm",
                                                           "dbscan", "optics", "mean_shift", "aggl", "aggl_features"],
                        help='Choose a clustering method', default="kmeans")
    parser.add_argument('--clustering_search_config', type=str,
                        help='Choose a clustering configuration file for the search',
                        default="../config/kmeans_search_test.json")
    parser.add_argument('--metric', type=str, choices=["silhouette"],
                        help='Choose a metric for measuring clustering performance', default="silhouette")
    parser.add_argument('--search_coverage', type=int, default=30, help='how many search configurations to try')

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
    parser.add_argument('--save_result_path', type=str, default=None,
                        help="where to salve the results, if necessary. By default results are not saved")
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    frac = 0.1 if args.test else 1
    n_jobs = 4 if args.test else -1

    data, names = load_sample_with_names(frac=frac, random_state=args.random_seed)

    with open(args.clustering_search_config) as f:
        clustering_search_config = json.load(f)
    with open(args.embedding_config) as f:
        embedding_args = json.load(f)

    clusterer = get_clusterer(args.clustering)
    embedder = get_embedder(args.embedding, random_state=args.random_seed, **embedding_args)

    if args.pre_filter:
        data, names = filter_twosides(data, names, get_twosides_meddra(False))

    if args.normalize:
        data = StandardScaler().fit_transform(data)

    if args.pre_embed:
        data = embedder.embed(data)

    search_result = ParamSearch(clusterer, clustering_search_config, args.metric)\
        .search(data, min_coverage=args.search_coverage, n_jobs=n_jobs)
    print(search_result[["params", "rank_test_score", "mean_test_score"]].sort_values("rank_test_score"))
    if args.save_result_path is not None:
        search_result.to_csv(args.save_result_path)
