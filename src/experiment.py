import os

from src.visualize.visualize import plot_embedded_cluster
from src.data.read_data import filter_twosides, get_twosides_meddra


class Experiment:
    def __init__(self, data, names, clusterer, embedder, pre_embedd=False, pre_filter=False, run_name="test"):
        self.data = data
        self.names = names
        self.clusterer = clusterer
        self.embedder = embedder
        self.pre_embedd = pre_embedd
        self.pre_filter = pre_filter
        self.run_path = os.path.join("../results", run_name)

    def run(self):
        data = self.data

        if self.pre_filter:
            data, names = filter_twosides(self.data, self.names, get_twosides_meddra())
        if self.pre_embedd:
            data = self.embedder.embed(self.data)

        self.clusterer.fit(data)

        if not self.pre_filter:
            data, names = filter_twosides(self.data, self.names, get_twosides_meddra())

        labels = self.clusterer.predict(data)

        if not self.pre_embedd:
            data = self.embedder.embed(self.data)

        print(labels)
        plot_embedded_cluster(data, labels, save_fig_path=os.path.join(self.run_path, "embedded_clusters.png"))
