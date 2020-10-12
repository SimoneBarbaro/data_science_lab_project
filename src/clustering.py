import pandas as pd
import SimpSOM as sps
from sklearn.cluster import KMeans
import numpy as np

def make_SOM(x,  y, data):
  net = sps.somNet(x, y, data, PBC=True)
  net.train(0.01, 20000)
  net.save('filename_weights')
  net.nodes_graph(colnum=0)
