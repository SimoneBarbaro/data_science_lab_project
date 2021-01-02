import argparse
import numpy as np
import json

from clustering.clustering import get_clusterer
from data.read_data import *
from dimensionality_reduction.embedding import get_embedder
from experiment.experiment import Experiment
from experiment.result_analysis import ResultAnalyzer
from experiment.statistical_analysis import StatisticalAnalyzer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=["spider", "tiger"],
                        help='Choose the dataset to work with', default="spider")
    parser.add_argument('--match_datasets', action='store_true', default=False,
                        help='Filter the data to only include the same drugs as the other dataset')
    parser.add_argument('--clustering', type=str, choices=["som_cluster", "kmeans", "dpgmm", "gmm",
                                                           "dbscan", "optics", "mean_shift", "aggl"],
                        help='Choose a clustering method', default="kmeans")
    parser.add_argument('--clustering_config', type=str,
                        help='Choose a clustering configuration file', default="../config/kmeans_10.json")

    parser.add_argument('--normalize', action='store_true', default=True,
                        help="add if you want to do normalize the columns of the target prediction dataset")
    parser.add_argument('--pre_embed', action='store_true', default=False,
                        help="add if you want to do the embedding before clustering")
    parser.add_argument('--pre_filter', action='store_true', default=True,
                        help="add to filter results before clustering based on twosides")

    parser.add_argument('--run_name', type=str, default="test",
                        help="name of the run, a folder with that name will be created in results to store all the "
                             "relevant results of the run")
    parser.add_argument('--random_seed', type=int, default=42, help="global seed for random functions")

    parser.add_argument('--level', type=str, choices=["all", "soc", "hlgt", "hlt", "pt"], default="all",
                        help='choose the medDRA level or "all"')
    parser.add_argument('--method', type=str, choices=["ranks", "scores"], default="ranks",
                        help='choose the quantity on which to evaluate statistical significance')
    parser.add_argument('--alpha', type=float, default=0.005,
                        help="significance level for Grubb's min/max test")
    parser.add_argument('--sort_by', type=str, choices=["rank", "grubbs", "tfidf_score"], default="rank",
                        help='choose the quantity on which to sort the results')

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    data, names = load_sample_with_names(dataset=args.dataset,
                                         random_state=args.random_seed,
                                         filtered=args.match_datasets,
                                         save=True)

    with open(args.clustering_config) as f:
        clustering_args = json.load(f)
    with open("../config/tsne_default.json") as f:
        embedding_args = json.load(f)

    if args.clustering == "som_cluster":
        clustering_args["train_data"] = data

    clusterer = get_clusterer(args.clustering, **clustering_args)
    embedder = get_embedder("tsne", random_state=args.random_seed, **embedding_args)
    analyzer = ResultAnalyzer(os.path.join("../results", args.run_name),
                              os.path.join("../results", args.run_name,
                                           "results.csv"))

    experiment = Experiment(data, names, clusterer, embedder, pre_embedd=args.pre_embed,
                            pre_filter=True,
                            visualize=False,
                            normalize=args.normalize, run_name=args.run_name)
    experiment.run()

    with open(os.path.join(experiment.run_path, "results_info.json")) as f:
        results_info = json.load(f)
    results_info["dataset"] = args.dataset
    results_info["match_datasets"] = args.match_datasets
    with open(os.path.join(experiment.run_path, "results_info.json"), "w") as f:
        f.write(json.dumps(results_info))

    analyzer.full_analysis()
    stat_analyzer = StatisticalAnalyzer(experiment.run_path, args.method, args.alpha, args.sort_by,
                                        save=True, print_output=False)

    stat_analyzer.full_analysis()
    stat_analyzer.summarize()
    stat_analyzer.full_comparison()

    results_file = os.path.join(experiment.run_path, "results.csv")
    results = pd.read_csv(results_file)
    meddra = get_twosides_meddra(False)
    results_meddra = match_meddra(results, meddra)

    for cluster in results_meddra["cluster"].drop_duplicates():
        results_filterd = results_meddra[(results_meddra["cluster"] == cluster)]
        filtered_names = results_filterd[["name1", "name2"]].drop_duplicates()

        tmp = names.reset_index().merge(filtered_names, on=["name1", "name2"], how='outer',
                                        indicator=True)
        tmp = tmp[tmp["_merge"] == "both"]
        interesting_indexes = tmp["index"]
        tmp_data = data.loc[interesting_indexes]
        tmp_data.reindex(tmp_data.median().sort_values()[::-1].index, axis=1).to_pickle(
            os.path.join(analyzer.analysis_dir,
                         "important_targets_{}.pkl.gz".format(cluster)))
