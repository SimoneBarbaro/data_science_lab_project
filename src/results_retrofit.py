import os
import json
import pandas as pd
from sklearn.metrics import silhouette_score

from data.read_data import load_full_matrix_with_names, get_twosides_meddra, filter_twosides, match_meddra

subfolders_with_paths = [f.path for f in os.scandir("../results/") if f.is_dir()]
meddra = get_twosides_meddra(False)
for dataset in ["spider", "tiger"]:
    for to_filter in [False]:
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

            analysis_dir = os.path.join(folder, "analysis")
            if not os.path.exists(analysis_dir):
                print(folder + " missing analysis")
            else:
                results_file = os.path.join(folder, "results.csv")
                results = pd.read_csv(results_file)
                results_meddra = match_meddra(results, meddra)
                try:
                    with open(os.path.join(folder, "results_info.json")) as f:
                        results_info = json.load(f)
                        dataset = results_info["dataset"]
                        match_datasets = results_info.get("match_datasets", False)
                except FileNotFoundError:
                    dataset = "spider"
                    match_datasets = False
                data, names = load_full_matrix_with_names(dataset, filtered=match_datasets)
                for cluster in results_meddra["cluster"].drop_duplicates():
                    if not os.path.exists(os.path.join(analysis_dir,
                                                       "important_targets_{}.pkl.gz".format(cluster))):
                        results_filterd = results_meddra[(results_meddra["cluster"] == cluster)]
                        filtered_names = results_filterd[["name1", "name2"]].drop_duplicates()

                        tmp = names.merge(filtered_names, how='outer', indicator=True)
                        tmp = tmp[tmp["_merge"] == "both"]

                        interesting_indexes = tmp.index
                        tmp_data = data.loc[interesting_indexes]
                        tmp_data.reindex(tmp_data.median().sort_values()[::-1].index, axis=1).to_pickle(
                            os.path.join(analysis_dir,
                                         "important_targets_{}.pkl.gz".format(cluster)))
                print(folder + " targets done")
