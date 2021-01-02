# Add here the relative path of the data files relative to the data directory directory. The data must be in csv format
ORIGINAL_SPIDER_DATA = "SPIDER_predictions_ML_anonymized.csv"
ORIGINAL_TIGER_DATA = "TIGER_prediction_GS_anonymized.csv"
RARE_TARGETS = "rare.targets.csv"

FILTERED_SPIDER_DATA = "spider_filtered_with_tiger.csv"
FILTERED_TIGER_DATA = "tiger_filtered_with_spider.csv"

FILTERED_INFIX = "_filtered"
NAMES_SUFFIX = "_names"

SPIDER_MATRIX_SAMPLE = "matrix_spider{}_{}_{}{}.pkl.gz"
SPIDER_MATRIX_FULL = "matrix_spider{}_full{}.pkl.gz"
TIGER_MATRIX_SAMPLE = "matrix_tiger{}_{}_{}{}.pkl.gz"
TIGER_MATRIX_FULL = "matrix_tiger{}_full{}.pkl.gz"
