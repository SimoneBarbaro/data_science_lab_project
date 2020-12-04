import os
import json
import pandas as pd
from sklearn.metrics import silhouette_score

from data.read_data import load_full_matrix_with_names, get_twosides_meddra, filter_twosides

subfolders_with_paths = [f.path for f in os.scandir("../results/") if f.is_dir()]
meddra = get_twosides_meddra(False)
for dataset in ["spider", "tiger"]:
    for to_filter in [False, True]:
        data, names = load_full_matrix_with_names(dataset, to_filter)
        data, names = filter_twosides(data, names, meddra)

        for folder in subfolders_with_paths:
            if not os.path.exists(os.path.join(folder, "results_info.json")):
                if folder.count("tiger") > 0 and dataset == "spider":
                    continue
                try:
                    results_file = os.path.join(folder, "results.csv")
                    results = pd.read_csv(results_file)
                    try:
                        if data.values.shape[0] != len(results["cluster"]):
                            if not to_filter:
                                continue
                            else:
                                raise ValueError
                        score = silhouette_score(data.values, results["cluster"])
                        with open(os.path.join(folder, "results_info.json"), "w") as f:
                            f.write(json.dumps({"dataset": dataset, "match_datasets": to_filter,
                                                "silhouette_score": score}))
                            print(folder + " results info saved successfully")
                    except ValueError:
                        print(folder + " was built on too old version of dataset")

                        with open(os.path.join(folder, "results_info.json"), "w") as f:
                            f.write(json.dumps({"dataset": dataset, "match_datasets": to_filter}))

                except FileNotFoundError:
                    print(folder + " results not found")
