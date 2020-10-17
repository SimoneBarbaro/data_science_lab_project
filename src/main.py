from src.read_data import *
from src.clustering import *
from src.visualize import *

def run_tsne():
    """
    Run tSNE with a sample of 10% of the original single-drug data.
    """
    # Read and sample the data, create the matrix (see read_data.py)
    data_sample = get_spider_data_sample(frac=0.1, random_state=1)
    data = create_matrix(data_sample)
    # Dimension reduction (see clustering.py)
    embedding = tsne_dimred(data, perplexity=40, n_jobs=4, random_state=3)
    
    # Matplotlib visualization
    import matplotlib.pyplot as plt
    plt.plot(embedding[:, 0], embedding[:, 1], '.')
    
    # Visualization with Pygal (see visualize.py)
    #visualize_pygal_scatter(embedding)

def run_umap():
    """
    Run UMAP with a sample of 10% of the original single-drug data.
    """
    # Read and sample the data, create the matrix (see read_data.py)
    data_sample = get_spider_data_sample(frac=0.1, random_state=1)
    data = create_matrix(data_sample)
    # Dimension reduction (see clustering.py)
    embedding = umap_dimred(data, n_neighbors=100, min_dist=0.0, random_state=2)
    
    # Matplotlib visualization
    import matplotlib.pyplot as plt
    plt.plot(embedding[:, 0], embedding[:, 1], '.')
    
    # Visualization with Pygal (see visualize.py)
    #visualize_pygal_scatter(embedding)

def run_kmeans_tsne():
    """
    Run k-means clustering with a sample of 10% of the original data,
    and visualize with tSNE.
    """
    # Read and sample the data, create the matrix (see read_data.py)
    data_sample = get_spider_data_sample(frac=0.10, random_state=1)
    data = create_matrix(data_sample)
    
    # Clustering
    from sklearn.cluster import KMeans
    k = 10
    kmeans = KMeans(n_clusters=k, random_state=4).fit(data)
    labels = kmeans.labels_
    
    # Dimension reduction (see clustering.py)
    embedding = tsne_dimred(data, perplexity=40, n_jobs=4, random_state=3)
    
    import matplotlib.pyplot as plt
    plt.title('Number of clusters: %d' % k)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=5, cmap='gist_ncar')

def run_kmeans_elbow():
    """
    Run k-means clustering with a sample of 10% of the original data,
    and plot the "elbow plot" for k=1,...,29.
    """
    # Read and sample the data, create the matrix (see read_data.py)
    data_sample = get_spider_data_sample(frac=0.10, random_state=1)
    data = create_matrix(data_sample)
    
    # Elbow plot
    from sklearn.cluster import KMeans
    sum_of_squared_distances = []
    K = range(1,30)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        sum_of_squared_distances.append(km.inertia_)
    
    import matplotlib.pyplot as plt
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('sum of squared distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
def run_dpgmm_tsne():
    """
    Runs a infinite gaussian mixture model with 5% of data
    """
    data_sample = get_spider_data_sample(frac=0.10, random_state=1)
    data = create_matrix(data_sample)
    X = data.values
    
    from sklearn import mixture
    dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                        covariance_type='full').fit(X)
    labels = dpgmm.predict(X)
    
    # Dimension reduction
    embedding = tsne_dimred(data, perplexity=40, n_jobs=4, random_state=3)
    
    import matplotlib.pyplot as plt
    plt.title('Number of clusters: %d' % k)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=5, cmap='gist_ncar')
    
def run_gmm_tsne(k = 8):
    """
    Runs a gaussian mixture model with 5% of data (this is what worked on local machine)
    """
    data_sample = get_spider_data_sample(frac=0.10, random_state=1)
    data = create_matrix(data_sample)
    
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=k,covariance_type='full').fit(data)
    labels = gmm.predict(data)
    
     # Dimension reduction (see clustering.py)
    embedding = tsne_dimred(data, perplexity=40, n_jobs=4, random_state=3)
    
    import matplotlib.pyplot as plt
    plt.title('Number of clusters: %d' % k)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=5, cmap='gist_ncar')