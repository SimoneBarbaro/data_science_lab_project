import os
import warnings
import numpy as np
import pandas as pd
from functools import reduce
from outliers import smirnov_grubbs as grubbs  # pip install outlier_utils

class StatisticalAnalyzer:

    def __init__(self, run_dir, method, alpha, sort_by, save=True, print_output=True):
        self.run_dir = run_dir
        self.analysis_dir = os.path.join(self.run_dir, "analysis")
        self.method = method
        self.alpha = alpha
        self.sort_by = sort_by
        self.save = save
        self.print_output = print_output  # only in code, not available in CLI; mainly used for full_analysis
        warnings.filterwarnings(
            "ignore")  # ignore Grubb's test "RuntimeWarning: invalid value encountered in double_scalars"

    def full_analysis(self):
        if self.save:
            self.print_output = False
        for level in ["soc", "hlgt", "hlt", "pt"]:
            self.analyze(level)

    def analyze(self, level):
        scores_file = os.path.join(self.analysis_dir, "scores_{}_term.csv".format(level))
        scores = pd.read_csv(scores_file)
        if not "rank" in scores.columns:
            ranked = scores.assign(
                rank=scores.groupby("cluster")["tfidf_score"].rank(method="average", ascending=False))
        else:
            ranked = scores

        term = "{}_term".format(level)

        significant = pd.DataFrame()
        if self.method == "ranks":
            for term in ranked.iloc[:, 1].unique():  # scores.iloc[:,1] corresponds to soc_term/pt_term/etc.
                ranks = ranked[ranked.iloc[:, 1] == term]["rank"]  # use ranks
                mean = np.mean(ranks)
                std_dev = np.std(ranks)
                sig_values = grubbs.min_test_outliers(list(ranks), alpha=self.alpha)
                for val in sig_values:
                    grubbs_statistic = (mean - val) / std_dev
                    significant = significant.append(
                        ranked[(ranked.iloc[:, 1] == term) & (ranked["rank"] == val)].assign(grubbs=grubbs_statistic))
        elif self.method == "scores":
            for term in ranked.iloc[:, 1].unique():  # scores.iloc[:,1] corresponds to soc_term/pt_term/etc.
                tfidfs = ranked[ranked.iloc[:, 1] == term]["tfidf_score"]  # use tf-idf scores
                mean = np.mean(tfidfs)
                std_dev = np.std(tfidfs)
                sig_values = grubbs.max_test_outliers(list(tfidfs), alpha=self.alpha)
                for val in sig_values:
                    grubbs_statistic = (val - mean) / std_dev
                    significant = significant.append(
                        ranked[(ranked.iloc[:, 1] == term) & (ranked["tfidf_score"] == val)].assign(
                            grubbs=grubbs_statistic))

        if self.sort_by == "rank":
            significant = significant.sort_values("rank", ascending=True)
        else:
            significant = significant.sort_values(self.sort_by, ascending=False)

        if self.print_output:
            print(significant.to_markdown(index=False))

        if self.save:
            results_file = os.path.join(self.analysis_dir,
                                        "significant_{}_{}_{}.csv".format(level, self.method, self.alpha))
            significant.to_csv(results_file, index=False)
    
    def summarize(self):
        results_drugs = pd.read_csv(os.path.join(self.run_dir, "results.csv"))
        nclusters = results_drugs["cluster"].nunique()
        signif = {
            "soc": pd.read_csv(os.path.join(self.analysis_dir, "significant_soc_{}_{}.csv".format(self.method, self.alpha))),
            "hlgt": pd.read_csv(os.path.join(self.analysis_dir, "significant_hlgt_{}_{}.csv".format(self.method, self.alpha))),
            "hlt": pd.read_csv(os.path.join(self.analysis_dir, "significant_hlt_{}_{}.csv".format(self.method, self.alpha))),
            "pt": pd.read_csv(os.path.join(self.analysis_dir, "significant_pt_{}_{}.csv".format(self.method, self.alpha)))
        }
        
        def get_top_n(signif, level, n):
            nonlocal nclusters
            return signif.groupby("cluster")["{}_term".format(level)]\
                         .nth(n).to_frame()\
                         .reindex(pd.Index(np.arange(0,nclusters), name="cluster"))\
                         .rename(columns={"{}_term".format(level):"{}{}".format(level, n+1)})
        
        # Top soc-level side effect (NaN if none)
        soc1 = get_top_n(signif["soc"], "soc", 0)
        result_dfs = [results_drugs, soc1]
        # Top side effects for other levels
        for level in ["hlgt", "hlt", "pt"]:
            for n in range(0,3):
                top_n = get_top_n(signif[level], level, n)
                result_dfs.append(top_n)

        summary = reduce(lambda left, right: pd.merge(left, right, on="cluster"), result_dfs)
        summary.to_csv(os.path.join(self.analysis_dir, "significant_summary.csv"), index=False)