import os
import numpy as np

from data.read_data import create_matrix, get_spider_data
from clustering.clustering import SomClusterer
from visualize.visualize import plot_embedded_cluster

if not os.path.exists("../data/matrix_spider.npy"):
    raw_data = create_matrix(get_spider_data().head(10)).values
    np.save("../data/matrix_spider.npy", raw_data)
else:
    raw_data = np.load("../data/matrix_spider.npy")
clusterer = SomClusterer(raw_data, size=20)


clusters = clusterer.get_som_clusters(raw_data, True)
print(clusters)
plot_embedded_cluster(clusterer.embed(raw_data), clusterer.get_cluster_labels(clusters))
