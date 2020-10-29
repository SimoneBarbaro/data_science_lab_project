import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

from data.read_data import get_twosides_meddra, match_meddra


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
