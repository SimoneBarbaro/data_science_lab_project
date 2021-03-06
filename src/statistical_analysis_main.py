import argparse
import os
from experiment.statistical_analysis import StatisticalAnalyzer

if __name__ == "__main__":

    # warnings.filterwarnings("ignore")  # ignore Grubb's test "RuntimeWarning: invalid value encountered in double_scalars"

    parser = argparse.ArgumentParser()
    # Sample command for copy-pasting and editing:
    # python ./statistical_analysis_main.py --run_name kmeans16 --level hlgt --method ranks --alpha 0.005 --sort_by rank --print_only
    # Sample command for running complete analysis for a run:
    # python ./statistical_analysis_main.py --run_name kmeans16 --level all --method ranks --alpha 0.005 --sort_by rank
    # Sample command for running the summary table:
    # python ./statistical_analysis_main.py --run_name kmeans16 --method ranks --alpha 0.005 --summarize
    # Sample command for running the comparison matrix:
    # python ./statistical_analysis_main.py --level all --method ranks --alpha 0.005 --compare

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
    parser.add_argument('--summarize', action='store_true', default=False,
                        help=("Add to run summary table (assumes analysis files for all levels). "
                              "Arguments 'level', 'sort_by', 'print_only' are ignored."))
    parser.add_argument('--compare', action='store_true', default=False,
                        help=("Add to run comparison matrix. "
                              "Arguments 'run_name', 'sort_by', 'print_only' are ignored."))

    args = parser.parse_args()

    run_dir = os.path.join("../results", args.run_name)

    analyzer = StatisticalAnalyzer(run_dir, args.method, args.alpha, args.sort_by, not args.print_only)
    
    if args.compare:
        if args.level == "all":
            analyzer.full_comparison()
        else:
            analyzer.compare(args.level)
    elif args.summarize:
        analyzer.summarize()
    elif args.level == "all":
        analyzer.full_analysis()
    else:
        analyzer.analyze(args.level)