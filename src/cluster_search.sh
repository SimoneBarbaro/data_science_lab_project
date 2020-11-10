#!/usr/bin/env bash
module load python/3.6.0 python/3.6.1
for clustering in "kmeans" "gmm" "dbscan" "optics" "mean_shift"; do
    bsub -W 12:00 -n 8 -R "rusage[mem=4096]" -oo ../results/${clustering}.txt python search_clustering_main.py --clustering ${clustering} --clustering_search_config ../config/${clustering}_search.json --save_result_path ../results/${clustering}.csv
    bsub -W 12:00 -n 8 -R "rusage[mem=4096]" -oo ../results/${clustering}_norm.txt python search_clustering_main.py --clustering ${clustering} --clustering_search_config ../config/${clustering}_search.json --normalize --save_result_path ../results/${clustering}_norm.csv
done

bsub -W 12:00 -n 8 -R "rusage[mem=4096]" -oo ../results/aggl1.txt python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_fixed_n.json --save_result_path ../results/aggl1.csv
bsub -W 12:00 -n 8 -R "rusage[mem=4096]" -oo ../results/aggl2.txt python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_fixed_n.json --normalize --save_result_path ../results/aggl2.csv
bsub -W 12:00 -n 8 -R "rusage[mem=4096]" -oo ../results/aggl3.txt python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_no_n.json --save_result_path ../results/aggl3.csv
bsub -W 12:00 -n 8 -R "rusage[mem=4096]" -oo ../results/aggl4.txt python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_no_n.json --normalize --save_result_path ../results/aggl4.csv