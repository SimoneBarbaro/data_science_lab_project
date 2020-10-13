import os
import numpy as np

from src.read_data import create_matrix, get_spider_data
from src.clustering import SomClusterer

if not os.path.exists("../data/matrix_spider.npy"):
    raw_data = create_matrix(get_spider_data().head(10)).values
    np.save("../data/matrix_spider.npy", raw_data)
else:
    raw_data = np.load("../data/matrix_spider.npy")
clusterer = SomClusterer(raw_data, size=20)

if not os.path.exists("../results/som.npy"):
    clusterer.train()
    clusterer.save("../results/som.npy")
else:
    clusterer.load("../results/som.npy")

clusters = clusterer.get_clusters(raw_data, True)
print(clusters)
