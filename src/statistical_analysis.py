import argparse
import os
import warnings
import numpy as np
import pandas as pd
from outliers import smirnov_grubbs as grubbs  # pip install outlier_utils
from experiment.statistical_analysis import StatisticalAnalyzer


def what(data, names, results_meddra, significant_clusters):
    term = significant_clusters.columns.drop(['value', 'tfidf_score', 'rank', 'grubbs', "cluster"])[0]
    for i, row in significant_clusters.drop(['value', 'tfidf_score', 'rank', 'grubbs'], axis=1).iterrows():
        filter = (results_meddra["cluster"] == row["cluster"]) & (results_meddra[term] == row[term])  # TODO term
        results_filterd = results_meddra[filter]
        filtered_names = results_filterd[["name1", "name2"]].drop_duplicates()
        """
        interesting_indexes = names[
                (names["name1"].isin(filtered_names["name1"])) & (names["name2"].isin(filtered_names["name2"]))].index
        """
        for i, names_row in filtered_names.iterrows():
            interesting_indexes = names[
                (names["name1"] == names_row["name1"]) & (names["name2"] == names_row["name2"])].index
            interesting_data = data.loc[interesting_indexes]
            interesting_data.head()
            print(interesting_data)


def get_more_significant_clusters(df, num_to_get=5):
    counts = df["cluster"].value_counts().sort_values(ascending=False)
    count_col = counts.loc[df["cluster"]].reset_index(drop=True)
    count_col.index = df.index
    df = df.assign(counts=count_col)

    clusters_to_get = counts.head(num_to_get).index
    df = df[df["cluster"].isin(clusters_to_get)]

    return df


if __name__ == "__main__":

    # warnings.filterwarnings("ignore")  # ignore Grubb's test "RuntimeWarning: invalid value encountered in double_scalars"

    parser = argparse.ArgumentParser()
    # Sample command for copy-pasting and editing:
    # python ./statistical_analysis.py --run_name kmeans16 --level hlgt --method ranks --alpha 0.005 --sort_by rank --print_only
    # Sample command for running complete analysis for a run:
    # python ./statistical_analysis.py --run_name kmeans16 --level all --method ranks --alpha 0.005 --sort_by rank

    parser.add_argument('--run_name', type=str, default="test",
                        help="name of the run where to look for the analysis folder with the scores")
    parser.add_argument('--level', type=str, choices=["all", "soc", "hlgt", "hlt", "pt"], default="hlgt",
                        help='choose the medDRA level or "all"')
    parser.add_argument('--method', type=str, choices=["ranks", "scores"], default="ranks",
                        help='choose the quantity on which to evaluate statistical significance')
    parser.add_argument('--alpha', type=float, default=0.005,
                        help="significance level for Grubb's min/max test")
    parser.add_argument('--sort_by', type=str, choices=["rank", "grubbs", "tfidf_score"], default="rank",
                        help='choose the quantity on which to sort the results')
    parser.add_argument('--print_only', action='store_true', default=False,
                        help="add to only print the results without saving")

    args = parser.parse_args()

    run_dir = os.path.join("../results", args.run_name)

    analyzer = StatisticalAnalyzer(run_dir, args.method, args.alpha, args.sort_by, not args.print_only)

    if args.level == "all":
        analyzer.full_analysis()
    else:
        analyzer.analyze(args.level)