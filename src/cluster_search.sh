#!/usr/bin/env bash
module load python/3.6.0 python/3.6.1
for clustering in "kmeans" "dpgmm" "gmm" "dbscan" "optics" "mean_shift"; do
    bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering ${clustering} --clustering_search_config ../config/${clustering}_search.json --save_result_path ../results/${clustering}.csv
    bsub -W 3:59 -n 8 -R "rusage[mem=4096]" python search_clustering_main.py --clustering ${clustering} --clustering_search_config ../config/${clustering}_search.json --normalize --save_result_path ../results/${clustering}.csv
done

bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_fixed_n.json --save_result_path ../results/aggl1.csv
bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_fixed_n.json --normalize --save_result_path ../results/aggl2.csv
bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_no_n.json --save_result_path ../results/aggl3.csv
bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering aggl --clustering_search_config ../config/agglomerative_search_no_n.json --normalize --save_result_path ../results/aggl4.csv
bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering aggl_features --clustering_search_config ../config/agglomerative_search_fixed_n.json --save_result_path ../results/aggl5.csv
bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering aggl_features --clustering_search_config ../config/agglomerative_search_fixed_n.json --normalize --save_result_path ../results/aggl6.csv
bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering aggl_features --clustering_search_config ../config/agglomerative_search_no_n.json --save_result_path ../results/aggl7.csv
bsub -W 3:59 -n 8 -R "rusage[mem=4096]" -oo ../results/tmp.txt python search_clustering_main.py --clustering aggl_features --clustering_search_config ../config/agglomerative_search_no_n.json --normalize --save_result_path ../results/aggl8.csv