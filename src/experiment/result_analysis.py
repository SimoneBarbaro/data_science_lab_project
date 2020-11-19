import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

from data.read_data import get_twosides_meddra, match_meddra
from sklearn.metrics.cluster import normalized_mutual_info_score

def intersect2D(a, b):
    """
    Find row intersection between 2D numpy arrays, a and b.
    Returns another numpy array with shared rows
    """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def max_swap(a):
    """
    Swaps to make the diagonal maximum
    """
    for i in range(min(a.shape)):
        sub = a[i:, i:]
        maxi = np.argmax(sub)
        ind1 = i + (maxi // (a.shape[1] - i)) % a.shape[0]
        ind2 = (maxi % (a.shape[1])) + i
        cpy2 = np.copy(a[:, i])
        a[:, i] = a[:, ind2]
        a[:, ind2] = cpy2
        cpy1 = np.copy(a[i, :])
        #print(cpy1, cpy2)
        a[i, :] = a[ind1, :]
        a[ind1, :] = cpy1
    return a
    

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

    def full_analysis(self):
        """
        Analyze the results and save the files in a subfolder in the run directory.
        """
        twosides = get_twosides_meddra(pickle=False)
        results = pd.read_csv(self.results_file)
        results_meddra = match_meddra(results, twosides)

        for term in ["soc_term", "hlgt_term", "hlt_term", "pt_term"]:
            self.analyze(results_meddra, meddra_term=term)

    def analyze(self, results_meddra, meddra_term="soc_term"):
        """
        Analyze the results for a single meddra term and save the files in a subfolder in the run directory.
        """
        scores_series = (results_meddra.groupby(["cluster", meddra_term]).size() / results_meddra.groupby(["cluster"]).size()) \
            .sort_values(ascending=False) \
            .sort_index(level=0, sort_remaining=False) \
            .groupby("cluster")

        scores_dataframe = scores_series.head(scores_series.size().sum()).to_frame("perc")

        tmp_df = scores_dataframe.reset_index()\
            .pivot(index="cluster", columns=meddra_term, values="perc")\
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
        final_dataframe = final_dataframe.assign(rank = final_dataframe.groupby("cluster").rank(method="average", ascending=False)["tfidf_score"])
        final_dataframe.to_csv(os.path.join(self.analysis_dir, "scores_{}.csv".format(meddra_term)), index=True, header=True)
        
    def cluster_intersection(self, results2):
        """
        Finds the intersection of two sets how many elements are shared between each cluster"
        """
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
        v1 = v1[ranks]
        v2 = v2[ranks2]
    
        m1 = mat/v1[:, None]
        m2 = mat/v2[None, :]
       
        m1 = max_swap(m1)
        m2 = max_swap(m2)
    
        return mat/v1[:, None], mat/v2[None, :]

    def mut_info(self, results_file2):
        """Finds the mutual info between two clustering methods"""
        a = pd.read_csv(results_file2)
        b = pd.read_csv(self.results_file)
        mutual_info = normalized_mutual_info_score(a["cluster"], b["cluster"])
        return mutual_info