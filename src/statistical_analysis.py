import argparse
import os
import warnings
import numpy as np
import pandas as pd
from outliers import smirnov_grubbs as grubbs  # pip install outlier_utils


def np_mad(arr, axis=None):
    """Median absolute deviation as alternative to np.std"""
    return 1.4826 * np.median(np.abs(arr - np.median(arr, axis)), axis)


def statistical_analysis(scores, level, method="rank", alpha=0.05, sort_by="rank"):
    ranked = scores.assign(rank=scores.groupby("cluster").rank(method="average", ascending=False)["tfidf_score"])

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

    significant = statistical_analysis(scores, level=args.level, method=args.method,
                                       alpha=args.alpha, sort_by=args.sort_by)

    if not args.print_only:
        results_file = os.path.join("../results", args.run_name, "analysis",
                                    "significant_{}_{}_{}_{}.csv".format(args.level, args.method,
                                                                         args.alpha, args.sort_by))
        significant.to_csv(results_file, index=False)

    print(significant.to_markdown(index=False))
