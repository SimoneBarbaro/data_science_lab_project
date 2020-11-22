import argparse
import os
from experiment.interactive_analysis import InteractiveAnalyzer
import pandas as pd

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--run_name', type=str, default="test", 
						help = "name of the file where to look for the analysis folder with the scores")
	parser.add_argument('--level', type=str, choices=["all", "soc", "hlgt", "hlt", "pt"], default="hlgt",
                        help="choose the medDRA level or 'all'")
	parser.add_argument('--num_to_get', type=int, default=5,
						help="Sets the number of significant clusters to get")
	parser.add_argument('--targets_per_cluster', type=int, default=5,
						help="Sets the number of targets to find per cluster")
	parser.add_argument('--all', action='store_true', default=False,
						help="Whether or not to analyze all files at once")

	args = parser.parse_args()
	
	if args.all:
		list_subfolders_with_paths = [f.path for f in os.scandir('../results/') if f.is_dir()]
		
		for pth in list_subfolders_with_paths:
			analyzer = InteractiveAnalyzer(pth)
			df1, df2 = analyzer.get_important_data(args.level, args.num_to_get, args.targets_per_cluster)
			df = pd.concat([pd.DataFrame(v) for k, v in df2.items()], axis = 1, keys = list(df2.keys()))
		
		targ_name = "targets_" + str(args.level) + ".csv"
		targets_dir = os.path.join("../results/analysis", targ_name)
		df.to_csv(os.path.join(analyzer.analysis_dir, targ_name), index = False)

	results_dir = os.path.join("../results", args.run_name)
	analyzer = InteractiveAnalyzer(results_dir)

	df1, df2 = analyzer.get_important_data(args.level, args.num_to_get, args.targets_per_cluster)
	df = pd.concat([pd.DataFrame(v) for k, v in df2.items()], axis = 1, keys = list(df2.keys()))

	targ_name = "targets_" + str(args.level) + ".csv"
	targets_dir = os.path.join("../results/analysis", targ_name)
	df.to_csv(os.path.join(analyzer.analysis_dir, targ_name), index = False)