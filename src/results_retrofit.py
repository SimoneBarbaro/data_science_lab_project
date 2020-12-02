import os
import json
import warnings
import numpy as np
import pandas as pd
from functools import reduce
from outliers import smirnov_grubbs as grubbs  # pip install outlier_utils
from sklearn.metrics import silhouette_score

from data.read_data import load_full_matrix_with_names, get_twosides_meddra, filter_twosides

subfolders_with_paths = [f.path for f in os.scandir("../results/") if f.is_dir()]
meddra = get_twosides_meddra(False)
data, names = load_full_matrix_with_names("spider")
data_spider, names_spider = filter_twosides(data, names, meddra)
data, names = load_full_matrix_with_names("tiger")
data_tiger, names_tiger = filter_twosides(data, names, meddra)
for folder in subfolders_with_paths:
    if folder.count("tiger") > 0:
        dataset = "tiger"
    else:
        dataset = "spider"

    results_file = os.path.join(folder, "results.csv")
    results = pd.read_csv(results_file)
    results
