#!/usr/bin/env bash
for clustering in "kmeans" "gmm" "dbscan" "optics" "mean_shift"; do
    python search_clustering_main.py --clustering ${clustering} --clustering_search_config ../config/${clustering}_search.json --save_result_path ../results/${clustering}.csv &
    python search_clustering_main.py --clustering ${clustering} --clustering_search_config ../config/${clustering}_search.json --normalize --save_result_path ../results/${clustering}_norm.csv &
done

python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_fixed_n.json --search_coverage 20 --save_result_path ../results/aggl1.csv &
python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_fixed_n.json --search_coverage 20 --normalize --save_result_path ../results/aggl2.csv &
python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_no_n.json --search_coverage 20 --save_result_path ../results/aggl3.csv &
python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_no_n.json --search_coverage 20 --normalize --save_result_path ../results/aggl4.csv &
