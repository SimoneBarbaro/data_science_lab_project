import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

from data.read_data import get_twosides_meddra, match_meddra

def intersect2D(a, b):
      """
      Find row intersection between 2D numpy arrays, a and b.
      Returns another numpy array with shared rows
      """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])
    
class ResultAnalyzer:
    """
    Analyzer for our clustering results.
    """
    def __init__(self, run_dir, results_file):
        """
        Create an Analyzer.
        :param run_dir: The folder where the run is saved.
        :param results_file: The results file name  # TODO in theory we could find this file automatially from run dir,
                # TODO but if we want to add more results to the folder this could break so I do it manually for now.
        """
        self.run_dir = run_dir
        self.analysis_dir = os.path.join(self.run_dir, "analysis")
        self.results_file = results_file
        os.makedirs(self.analysis_dir, exist_ok=True)

    def analyze(self, meddra_term="soc_term"):
        """
        Analyze the results and save the files in a subfolder in the run directory.
        """
        twosides = get_twosides_meddra(pickle=False)
        kmeans_results = pd.read_csv(self.results_file)
        kmeans_meddra = match_meddra(kmeans_results, twosides)

        scores_series = (kmeans_meddra.groupby(["cluster", meddra_term]).size() / kmeans_meddra.groupby(["cluster"]).size()) \
            .sort_values(ascending=False) \
            .sort_index(level=0, sort_remaining=False) \
            .groupby("cluster")

        scores_dataframe = scores_series.head(scores_series.size().sum()).to_frame("value")

        tmp_df = scores_dataframe.reset_index()\
            .pivot(index="cluster", columns=meddra_term, values="value")\
            .fillna(0)
        tfidf = TfidfTransformer()
        values = tfidf.fit_transform(tmp_df).toarray()

        tfidf_series = pd.DataFrame(values, columns=tmp_df.columns, index=tmp_df.index)\
            .stack()\
            .sort_values(ascending=False)\
            .sort_index(level=0, sort_remaining=False)
        tfidf_series = tfidf_series[tfidf_series > 0]

        tfidf_dataframe = tfidf_series.to_frame("tfidf_score")

        final_dataframe = pd.concat([scores_dataframe, tfidf_dataframe], axis=1)
        final_dataframe.to_csv(os.path.join(self.analysis_dir, "scores_{}.csv".format(meddra_term)), index=True, header=True)
    
    def cluster_intersection(self, results2):
        size = []
        assignments = []
        for i in range(len(self.results_file["cluster"].unique())):
            size.append(self.results_file[self.results_file["cluster"] == i].count()["cluster"])
            assignments.append(self.results_file[self.results_file["cluster"] == i][self.results_file.columns[0:2]].values)
    
         assignments_ordered = []
        ranks = np.argsort(size)
        for j in range(len(size)):
            assignments_ordered.append(assignments[ranks[j]])
    
        size2 = []
        assignments2 = []
        for i in range(len(results2["cluster"].unique())):
            size2.append(results2[results2["cluster"] == i].count()["cluster"])
            assignments2.append(results2[results2["cluster"] == i][results2.columns[0:2]].values)
    
        assignments_ordered2 = []
        ranks2 = np.argsort(size2)
        for j in range(len(size2)):
            assignments_ordered2.append(assignments2[ranks2[j]])
    
        mat = np.zeros((len(assignments_ordered), len(assignments_ordered2)))
    
        for i in range(len(assignments_ordered)):
            for j in range(len(assignments_ordered2)):
                mat[i, j] = len(intersect2D(assignments_ordered[i], assignments_ordered2[j]))
    
        v1 = np.array(size)
        v2 = np.array(size2)
    
        return mat/v1[:, None], mat/v2[None, :]