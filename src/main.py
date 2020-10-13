from src.read_data import *
from src.clustering import *
from src.visualize import *

def run_tsne():
    """
    Run TSNE with a sample of 10% of the original single-drug data.
    """
    # Read and sample the data, create the matrix (see read_data.py)
    data_sample = get_spider_data_sample(frac=0.1)
    data = create_matrix(data_sample)
    # Clustering (see clustering.py)
    embedded = tsne_dimred(data, n_jobs=4)
    
    # Matplotlib visualization
    import matplotlib.pyplot as plt
    plt.plot(embedded[:, 0], embedded[:, 1], 'o')
    
    # Visualization with Pygal (see visualize.py)
    #visualize_pygal_scatter(embedded)