import os

from src.visualize.visualize import plot_embedded_cluster


class Experiment:
    def __init__(self, data, clusterer, embedder, pre_embedd=False, run_name="test"):
        self.data = data
        self.clusterer = clusterer
        self.embedder = embedder
        self.pre_embedd = pre_embedd
        self.run_path = os.path.join("../results", run_name)

    def run(self):
        data = self.data
        if self.pre_embedd:
            data = self.embedder.embedd(self.data)
        labels = self.clusterer.fit(data).predict(data)
        if not self.pre_embedd:
            data = self.embedder.embedd(self.data)
        plot_embedded_cluster(data, labels, save_fig_path=os.path.join(self.run_path, "embedded_clusters.png"))
