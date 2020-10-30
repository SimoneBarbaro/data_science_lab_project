from clustering.clustering import get_clusterer
from data.read_data import *
from dimensionality_reduction.embedding import get_embedder
from experiment import Experiment
from result_analysis import ResultAnalyzer

import argparse
import json
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--clustering', type=str, choices=["som_cluster", "kmeans", "dpgmm", "gmm", "dbscan", "aggl"],
                        help='Choose a clustering method', default="kmeans")
    parser.add_argument('--clustering_config', type=str,
                        help='Choose a clustering configuration file', default="../config/kmeans_default.json")

    parser.add_argument('--embedding', type=str, choices=["tsne", "umap", "som"],
                        help='Choose an embedding method', default="tsne")
    parser.add_argument('--embedding_config', type=str,
                        help='Choose a embedding configuration file', default="../config/tsne_default.json")

    parser.add_argument('--pre_embed', action='store_true', default=False,
                        help="add if you want to do the embedding before clustering")
    parser.add_argument('--visualize', action='store_true', default=False,
                        help="add if you want to visualize the embeddings")
    parser.add_argument('--pre_filter', action='store_true', default=False,
                        help="add to filter results before clustering based on twosides")

    parser.add_argument('--run_name', type=str, default="test",
                        help="name of the run, a folder with that name will be created in results to store all the "
                             "relevant results of the run")
    parser.add_argument('--analysis', type=str, default="yes", choices=["yes", "no", "only"],
                        help="Options for running the analysis, yes for doing it after the run, "
                             "only for doing only it, use it if the run is already available")
    parser.add_argument('--random_seed', type=int, default=42, help="global seed for random functions")
    parser.add_argument('--test', action='store_true', default=False,
                        help="add if you want to test the run on a smaller amount of data")

    args = parser.parse_args()

    np.random.seed(args.random_seed)
    frac = 0.1 if args.test else 1

    data, names = load_sample_with_names(frac=frac, random_state=args.random_seed)

    with open(args.clustering_config) as f:
        clustering_args = json.load(f)
    with open(args.embedding_config) as f:
        embedding_args = json.load(f)

    if args.clustering == "som_cluster":
        clustering_args["train_data"] = data

    clusterer = get_clusterer(args.clustering, random_state=args.random_seed, **clustering_args)
    embedder = get_embedder(args.embedding, random_state=args.random_seed, **embedding_args)
    analyzer = ResultAnalyzer(os.path.join("../results", args.run_name),
                              os.path.join("../results", args.run_name,
                                           "results.csv"))
    if args.analysis == "only":
        analyzer.full_analysis()
    else:
        experiment = Experiment(data, names, clusterer, embedder, pre_embedd=args.pre_embed,
                                pre_filter=args.pre_filter, visualize=args.visualize,
                                run_name=args.run_name)
        experiment.run()
        if args.analysis == "yes":
            analyzer.full_analysis()
