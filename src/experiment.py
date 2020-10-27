import os
import pandas as pd

from src.visualize.visualize import plot_embedded_cluster
from src.data.read_data import filter_twosides, get_twosides_meddra


class Experiment:
    """
    Main class to run a clustering experiment. It is generic
    and should be able to run with any clustering and any dimensionality reduction embedding method.
    """
    def __init__(self, data, names, clusterer, embedder, pre_embedd=False, pre_filter=False, run_name="test"):
        """
        Create a new experiment.
        :param data: The data to run the experiment.
        :param names: The dataframe with the names of the pairs related to the data
        :param clusterer: A clusterer object, able to implement an unsupervised fit and predict clustering.
        :param embedder: An embedder object, able to embed data into a two dimensional space.
        :param pre_embedd: Whether to embed the data before clustering or after.
        :param pre_filter: whether to filter the data based on the twosides dataset before or after clustering and embedding.
        :param run_name: The name of the run corresponding to the folder in results where results will be stored.
        """
        self.data = data
        self.names = names
        self.clusterer = clusterer
        self.embedder = embedder
        self.pre_embedd = pre_embedd
        self.pre_filter = pre_filter
        self.run_path = os.path.join("../results", run_name)

    def run(self):
        """
        Run the experiment. All the results should be either printed or saved into the subfolder of results
        specified at initialization of the class.
        """
        data = self.data

        if self.pre_filter:
            data, names = filter_twosides(self.data, self.names, get_twosides_meddra(False))
        if self.pre_embedd:
            data = self.embedder.embed(data)

        self.clusterer.fit(data)

        if not self.pre_filter:
            data, names = filter_twosides(self.data, self.names, get_twosides_meddra(False))

        labels = self.clusterer.predict(data)

        if not self.pre_embedd:
            data = self.embedder.embed(data)

        results = pd.DataFrame(labels, columns=["cluster"])
        results = pd.concat([names.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
        print(results)
        results.to_csv(os.path.join(self.run_path, "results.csv"), index=False, header=True)
        os.makedirs(self.run_path, exist_ok=True)
        plot_embedded_cluster(data, labels, save_fig_path=os.path.join(self.run_path, "embedded_clusters.png"))
