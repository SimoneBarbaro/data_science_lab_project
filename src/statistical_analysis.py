import argparse
import os
import warnings
import numpy as np
import pandas as pd
from outliers import smirnov_grubbs as grubbs  # pip install outlier_utils


def np_mad(arr, axis=None):
    """Median absolute deviation as alternative to np.std"""
    return 1.4826 * np.median(np.abs(arr - np.median(arr, axis)), axis)


def what(data, names, results_meddra, significant_clusters):
    term = significant_clusters.columns.drop(['value', 'tfidf_score', 'rank', 'grubbs', "cluster"])[0]
    for i, row in significant.drop(['value', 'tfidf_score', 'rank', 'grubbs'], axis=1).iterrows():
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


def statistical_analysis(scores, method="rank", alpha=0.05, sort_by="rank"):
    ranked = scores.assign(rank=scores.groupby("cluster")["tfidf_score"].rank(method="average", ascending=False))

    significant = pd.DataFrame()
    scores_name = "rank" if method == "ranks" else "tfidf_score"
    for term in ranked.iloc[:, 1].unique():  # scores.iloc[:,1] corresponds to soc_term/pt_term/etc.
        ranks = ranked[ranked.iloc[:, 1] == term][scores_name]
        mean = np.mean(ranks)
        std_dev = np.std(ranks)
        sig_values = grubbs.min_test_outliers(list(ranks), alpha=alpha)
        for val in sig_values:
            grubbs_statistic = (mean - val) / std_dev
            significant = significant.append(
                ranked[(ranked.iloc[:, 1] == term) & (ranked[scores_name] == val)].assign(grubbs=grubbs_statistic))

    ascending = False
    if sort_by == "rank":
        ascending = True

    significant = significant.sort_values(sort_by, ascending=ascending)

    return significant


if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore")  # ignore Grubb's test "RuntimeWarning: invalid value encountered in double_scalars"

    parser = argparse.ArgumentParser()
    # Sample command for copy-pasting and editing:
    # python ./statistical_analysis.py --run_name kmeans16 --level hlgt --method ranks --alpha 0.005 --sort_by rank --print_only

    parser.add_argument('--run_name', type=str, default="test",
                        help="name of the run where to look for the analysis folder with the scores")
    parser.add_argument('--level', type=str, choices=["soc", "hlgt", "hlt", "pt"], default="hlgt",
                        help='choose the medDRA level')
    parser.add_argument('--method', type=str, choices=["ranks", "scores"], default="ranks",
                        help='choose the quantity on which to evaluate statistical significance')
    parser.add_argument('--alpha', type=float, default=0.005,
                        help="significance level for Grubb's min/max test")
    parser.add_argument('--sort_by', type=str, choices=["rank", "grubbs", "tfidf_score"], default="rank",
                        help='choose the quantity on which to sort the results')
    parser.add_argument('--print_only', action='store_true', default=False,
                        help="add to only print the results without saving")

    args = parser.parse_args()
    scores_file = os.path.join("../results", args.run_name, "analysis", "scores_{}_term.csv".format(args.level))
    scores = pd.read_csv(scores_file)

    significant = statistical_analysis(scores, method=args.method,
                                       alpha=args.alpha, sort_by=args.sort_by)

    if not args.print_only:
        results_file = os.path.join("../results", args.run_name, "analysis",
                                    "significant_{}_{}_{}_{}.csv".format(args.level, args.method,
                                                                         args.alpha, args.sort_by))
        significant.to_csv(results_file, index=False)

    print(significant.to_markdown(index=False))
