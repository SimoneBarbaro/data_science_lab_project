import json
import os

import pandas as pd
from data.read_data import get_twosides_meddra, match_meddra, load_full_matrix_with_names, get_rare_targets
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

        twosides = get_twosides_meddra(pickle=False)
        results = pd.read_csv(self.results_file)
        self.results_meddra = match_meddra(results, twosides)
        try:
            with open(os.path.join(results_dir, "results_info.json")) as f:
                results_info = json.load(f)
                dataset = results_info["dataset"]
                match_datasets = results_info.get("match_datasets", default=False)
        except FileNotFoundError:
            dataset = "spider"
            match_datasets = False
        self.data, self.names = load_full_matrix_with_names(dataset, filtered=match_datasets)

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
        results_filterd = self.results_meddra[(self.results_meddra["cluster"] == cluster)]
        filtered_names = results_filterd[["name1", "name2"]].drop_duplicates()

        tmp = self.names.merge(filtered_names, how='outer', indicator=True)
        tmp = tmp[tmp["_merge"] == "both"]

        interesting_indexes = tmp.index
        tmp_data = self.data.loc[interesting_indexes]

        return tmp_data.reindex(tmp_data.mean().sort_values()[::-1].index, axis=1).iloc[:, : targets_per_cluster]

    def get_important_data(self, level, cluster_number=5, targets_per_cluster=5):
        significant_clusters = self.get_more_significant_clusters(level, num_to_get=cluster_number)
        important_targets = {}
        for cluster in significant_clusters["cluster"].drop_duplicates():
            important_targets[cluster] = self.get_important_targets(cluster, targets_per_cluster)
        return significant_clusters, important_targets

    def get_rare_important_targets(self, cluster_number=5, targets_per_cluster=5):
        targets = get_rare_targets()
        hd = np.array(targets.columns)
        rare_clusters = []
        for cluster in range(cluster_number):
            important_targets = self.get_important_targets(cluster_number, targets_per_cluster)
            cluster_targets = important_targets.columns
            if pd.Series(hd).isin(cluster_targets).any():
                rare_clusters.append(cluster)
        return rare_clusters
