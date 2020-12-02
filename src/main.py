import argparse
import numpy as np
import json

from clustering.clustering import get_clusterer
from data.read_data import *
from dimensionality_reduction.embedding import get_embedder
from experiment.experiment import Experiment
from experiment.result_analysis import ResultAnalyzer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=["spider", "tiger"],
                        help='Choose the dataset to work with', default="spider")
    parser.add_argument('--match_datasets', action='store_true', default=False,
                        help='Filter the data to only include the same drugs as the other dataset')
    parser.add_argument('--clustering', type=str, choices=["som_cluster", "kmeans", "dpgmm", "gmm",
                                                           "dbscan", "optics", "mean_shift", "aggl", "aggl_features"],
                        help='Choose a clustering method', default="kmeans")
    parser.add_argument('--clustering_config', type=str,
                        help='Choose a clustering configuration file', default="../config/kmeans_default.json")

    parser.add_argument('--embedding', type=str, choices=["tsne", "umap", "som"],
                        help='Choose an embedding method', default="tsne")
    parser.add_argument('--embedding_config', type=str,
                        help='Choose a embedding configuration file', default="../config/tsne_default.json")

    parser.add_argument('--normalize', action='store_true', default=False,
                        help="add if you want to do normalize the columns of the target prediction dataset")
    parser.add_argument('--pre_embed', action='store_true', default=False,
                        help="add if you want to do the embedding before clustering")
    parser.add_argument('--visualize', action='store_true', default=False,
                        help="add if you want to visualize the embeddings")
    parser.add_argument('--pre_filter', action='store_true', default=False,
                        help="add to filter results before clustering based on twosides")

    parser.add_argument('--run_name', type=str, default="test",
                        help="name of the run, a folder with that name will be created in results to store all the "
                             "relevant results of the run")
    parser.add_argument('--analysis', type=str, default="yes", choices=["yes", "no", "only", "mutual"],
                        help="Options for running the analysis, yes for doing it after the run, "
                             "only for doing only it, use it if the run is already available")
    parser.add_argument('--random_seed', type=int, default=42, help="global seed for random functions")
    parser.add_argument('--test', action='store_true', default=False,
                        help="add if you want to test the run on a smaller amount of data")

    args = parser.parse_args()

    np.random.seed(args.random_seed)
    frac = 0.1 if args.test else 1

    data, names = load_sample_with_names(dataset=args.dataset,
                                         frac=frac,
                                         random_state=args.random_seed,
                                         filtered=args.match_datasets,
                                         save=True)

    with open(args.clustering_config) as f:
        clustering_args = json.load(f)
    with open(args.embedding_config) as f:
        embedding_args = json.load(f)

    if args.clustering == "som_cluster":
        clustering_args["train_data"] = data

    clusterer = get_clusterer(args.clustering, **clustering_args)
    embedder = get_embedder(args.embedding, random_state=args.random_seed, **embedding_args)
    analyzer = ResultAnalyzer(os.path.join("../results", args.run_name),
                              os.path.join("../results", args.run_name,
                                           "results.csv"))

    if args.analysis == "only":
        analyzer.full_analysis()
        #analyzer.mut_info('../results/kmeans_10')
    elif args.analysis == "mutual":
        list_subfolders_with_paths = [f.path for f in os.scandir('../results/') if f.is_dir()]
        print(list_subfolders_with_paths)
        mat = np.zeros((len(list_subfolders_with_paths), len(list_subfolders_with_paths)))
        for i in range(len(list_subfolders_with_paths)):
            analyze1 = ResultAnalyzer(list_subfolders_with_paths[i], os.path.join(list_subfolders_with_paths[i], 'results.csv'))
            for j in range(len(list_subfolders_with_paths)):
                mat[i, j] = analyze1.mut_info(os.path.join(list_subfolders_with_paths[j], 'results.csv'))
        df = pd.DataFrame(mat)
        df.index = list_subfolders_with_paths
        df.columns = list_subfolders_with_paths
        df.to_csv('../results/mutual_analysis.csv')
    else:
        experiment = Experiment(data, names, clusterer, embedder, pre_embedd=args.pre_embed,
                                pre_filter=True,  # args.pre_filter, TODO hotfix to not forget
                                visualize=args.visualize,
                                normalize=args.normalize, run_name=args.run_name)
        experiment.run()
        if args.analysis == "yes":
            analyzer.full_analysis()
