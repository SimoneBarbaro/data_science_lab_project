import os
import pandas as pd

from visualize.visualize import plot_embedded_cluster
from data.read_data import filter_twosides, get_twosides_meddra


class Experiment:
    """
    Main class to run a clustering experiment. It is generic
    and should be able to run with any clustering and any dimensionality reduction embedding method.
    """
    def __init__(self, data, names, clusterer, embedder,
                 pre_embedd=False, visualize=False, pre_filter=False, run_name="test"):
        """
        Create a new experiment.
        :param data: The data to run the experiment.
        :param names: The dataframe with the names of the pairs related to the data
        :param clusterer: A clusterer object, able to implement an unsupervised fit and predict clustering.
        :param embedder: An embedder object, able to embed data into a two dimensional space.
        :param pre_embedd: Whether to embed the data before clustering or after.
        :param visualize: whether to visualize the embeddings.
        :param pre_filter: whether to filter the data based on the twosides dataset before or after clustering and embedding.
        :param run_name: The name of the run corresponding to the folder in results where results will be stored.
        """
        self.data = data
        self.names = names
        self.clusterer = clusterer
        self.embedder = embedder
        self.pre_embedd = pre_embedd
        self.pre_filter = pre_filter
        self.visualize = visualize
        self.run_path = os.path.join("../results", run_name)
        self.filtered_data, self.filtered_names = filter_twosides(self.data, self.names, get_twosides_meddra(False))

    def get_train_data(self):
        """
        :return: The dataset for training the clusterer.
        """
        if self.pre_filter:
            if self.pre_embedd:
                return self.embedder.embed(self.filtered_data)
            return self.filtered_data
        if self.pre_embedd:
            return self.embedder.embed(self.data)
        return self.data

    def get_test_data(self):
        """
        :return: The dataset for testing and generating results on the clusters.
        """
        if self.pre_embedd:
            return self.embedder.embed(self.filtered_data)
        return self.filtered_data

    def predict_labels(self):
        """
        Fit the clusterer and predict cluster labels
        :return: predicted cluster labels
        """
        self.clusterer.fit(self.get_train_data())
        return self.clusterer.predict(self.get_test_data())

    def generate_results(self, labels):
        """
        Generate experiment results given predicted cluster labels
        :param labels: cluster labels
        """
        results = pd.DataFrame(labels, columns=["cluster"])
        results = pd.concat([self.filtered_names.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
        print(results)
        os.makedirs(self.run_path, exist_ok=True)
        results.to_csv(os.path.join(self.run_path, "results.csv"), index=False, header=True)

        if self.visualize:
            self.visualize_embeddings(labels)

    def visualize_embeddings(self, labels):
        """
        Visualize the embeddings for a given labels
        :param labels: cluster labels
        """
        if not self.pre_embedd:
            data = self.embedder.embed(self.get_test_data())
        else:
            data = self.get_test_data()
        plot_embedded_cluster(data, labels, save_fig_path=os.path.join(self.run_path, "embedded_clusters.png"))

    def run(self):
        """
        Run the experiment. All the results should be either printed or saved into the subfolder of results
        specified at initialization of the class.
        """
        labels = self.predict_labels()
        self.generate_results(labels)
