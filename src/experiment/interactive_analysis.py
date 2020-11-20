import os

import pandas as pd
from data.read_data import get_twosides_meddra, match_meddra, load_sample_with_names





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

class InteractiveAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.results_file = os.path.join(results_dir, "results.csv")
        self.analysis_dir = os.path.join(results_dir, "analysis")
        self.scores = {}
        self.significant = {}
        for level in ["soc", "pt", "hlt", "hlgt"]:
            path = os.path.join(self.analysis_dir, "scores_{}_term.csv".format(level))
            if os.path.exists(path):
                self.scores[level] = pd.read_csv(path)
            for file in os.listdir(self.analysis_dir):
                if file.startswith("significant_{}".format(level)):
                    path = os.path.join(self.analysis_dir, file)
                    self.significant[level] = pd.read_csv(path)

        twosides = get_twosides_meddra(pickle=False)
        results = pd.read_csv(self.results_file)
        self.results_meddra = match_meddra(results, twosides)
        self.data, self.names = load_sample_with_names()

    def get_more_significant_clusters(self, level, num_to_get=5):
        df = self.significant[level]
        counts = df["cluster"].value_counts().sort_values(ascending=False)
        count_col = counts.loc[df["cluster"]].reset_index(drop=True)
        count_col.index = df.index
        df = df.assign(counts=count_col)

        clusters_to_get = counts.head(num_to_get).index
        df = df[df["cluster"].isin(clusters_to_get)]

        return df

    def prova(self):
        self.result_analyzer.