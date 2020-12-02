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
    if not os.path.exists(os.path.join(folder, "results_info.json")):
        if folder.count("tiger") > 0:
            dataset = "tiger"
        else:
            dataset = "spider"
        try:
            results_file = os.path.join(folder, "results.csv")
            results = pd.read_csv(results_file)
            try:
                score = silhouette_score(data.values, results["cluster"])
                with open(os.path.join(folder, "results_info.json"), "w") as f:
                    f.write(json.dumps({"dataset": dataset, "silhouette_score": score}))
                    print(folder + " results info saved successfully")
            except ValueError:
                print(folder + " was built on too old version of dataset")
        except FileNotFoundError:
            print(folder + " results not found")
