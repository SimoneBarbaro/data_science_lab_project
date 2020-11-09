import argparse
import os
import numpy as np
import pandas as pd

def np_mad(arr, axis=None):
    """Median absolute deviation as alternative to np.std"""
    return 1.4826*np.median(np.abs(arr - np.median(arr, axis)), axis)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Sample command for copy-pasting and editing:
    # python ./statistical_analysis.py --run_name kmeans16 --level hlgt --method ranks --location mean --scatter std --distance 2.5 --print_only
    
    parser.add_argument('--run_name', type=str, default="test",
                        help="name of the run where to look for the analysis folder with the scores")
    parser.add_argument('--level', type=str, choices=["soc", "hlgt", "hlt", "pt"], default="hlgt",
                        help='choose the medDRA level')
    parser.add_argument('--method', type=str, choices=["ranks", "scores"], default="ranks",
                        help='choose the quantity on which to evaluate statistical significance')
    parser.add_argument('--location', type=str, choices=["mean", "median"], default="mean",
                        help='choose the location statistic')
    parser.add_argument('--scatter', type=str, choices=["std", "mad"], default="std",
                        help='choose the scatter statistic')
    parser.add_argument('--distance', type=float, default=3.0,
                        help="how many scatters from the location is considered significant")
    parser.add_argument('--print_only', action='store_true', default=False,
                        help="add to only print the results without saving")

    args = parser.parse_args()
    
    scores_file = os.path.join("../results", args.run_name, "analysis", "scores_{}_term.csv".format(args.level))
    scores = pd.read_csv(scores_file)
    ranked = scores.assign(rank = scores.groupby("cluster").rank(method="average", ascending=False)["tfidf_score"])
    
    term = "{}_term".format(args.level)
    
    location_method = lambda: None
    if args.location == "mean":
        location_method = np.mean
    elif args.location == "median":
        location_method = np.median
    
    scatter_method = lambda: None
    if args.scatter == "std":
        scatter_method = np.std
    elif args.scatter == "mad":
        scatter_method = np_mad
    
    significant = pd.DataFrame()
    if args.method == "ranks":
        for term in ranked.iloc[:,1].unique():  # scores.iloc[:,1] corresponds to soc_term/pt_term/etc.
            ranks = ranked[ranked.iloc[:,1] == term]["rank"]  # use ranks
            location = location_method(ranks)
            scatter = scatter_method(ranks)
            significant = significant.append(ranked[(ranked.iloc[:,1] == term) & (ranked["rank"] < location - args.distance*scatter)])
    elif args.method == "scores":
        for term in ranked.iloc[:,1].unique():  # scores.iloc[:,1] corresponds to soc_term/pt_term/etc.
            tfidfs = ranked[ranked.iloc[:,1] == term]["tfidf_score"]  # use tf-idf scores
            location = location_method(tfidfs)
            scatter = scatter_method(tfidfs)
            significant = significant.append(ranked[(ranked.iloc[:,1] == term) & (ranked["tfidf_score"] > location + args.distance*scatter)])
    
    if not args.print_only:
        results_file = os.path.join("../results", args.run_name, "analysis", "significant_{}_{}_{}_{}_{}.csv".format(args.level, args.method, args.location, args.scatter, args.distance))
        significant.to_csv(results_file, index=False)
    
    print(significant.to_markdown(index=False))