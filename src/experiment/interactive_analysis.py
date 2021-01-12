import os

import pandas as pd
from data.read_data import get_rare_targets
import numpy as np


class InteractiveAnalyzer:
    LEVELS = ["soc", "pt", "hlt", "hlgt"]

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.results_file = os.path.join(results_dir, "results.csv")
        self.analysis_dir = os.path.join(results_dir, "analysis")
        self.scores = {}
        self.significant_clusters = {}
        for level in self.LEVELS:
            path = os.path.join(self.analysis_dir, "scores_{}_term.csv".format(level))
            if os.path.exists(path):
                self.scores[level] = pd.read_csv(path)
            for file in os.listdir(self.analysis_dir):
                if file.startswith("significant_{}".format(level)):
                    path = os.path.join(self.analysis_dir, file)
                    self.significant_clusters[level] = pd.read_csv(path)

    def get_more_significant_clusters(self, level, num_to_get=5):
        df = self.significant_clusters[level]
        counts = df["cluster"].value_counts().sort_values(ascending=False)
        count_col = counts.loc[df["cluster"]].reset_index(drop=True)
        count_col.index = df.index
        df = df.assign(counts=count_col)

        clusters_to_get = counts.head(num_to_get).index
        df = df[df["cluster"].isin(clusters_to_get)]

        return df

    def get_important_targets(self, cluster, targets_per_cluster=5):
        cluster = float(cluster)
        return pd.read_pickle(os.path.join(self.analysis_dir, "important_targets_{}.pkl.gz".format(int(cluster)))).iloc[:, : targets_per_cluster]

    def get_important_data(self, level, cluster_number=5, targets_per_cluster=5):
        significant_clusters = self.get_more_significant_clusters(level, num_to_get=cluster_number)
        important_targets = {}
        for cluster in significant_clusters["cluster"].drop_duplicates():
            important_targets[cluster] = self.get_important_targets(cluster, targets_per_cluster)
        return significant_clusters, important_targets

    def get_rare_important_targets(self, cluster_number=5, targets_per_cluster=5):
        targets = get_rare_targets()
        hd = targets
        rare_clusters = []
        for cluster in range(cluster_number):
            important_targets = self.get_important_targets(cluster, targets_per_cluster)
            cluster_targets = important_targets.columns
            rare_clusters.append(hd.isin(cluster_targets).values.reshape(-1))
        result = pd.DataFrame(rare_clusters, columns=hd.values.reshape(-1))
        result.insert(0, "has_important_rare_targets", result.any(axis=1))
        return result
