"""
This file is for functions that search the best configuration of clustering parameters, like finding the k on kmeans.
Only methods that require human supervision should be here,
automatic methods should be incorporated into the fit method of a clusterer instead.
"""
from src.data.read_data import load_sample


def run_kmeans_elbow():
    """
    Run k-means clustering with a sample of 10% of the original data,
    and plot the "elbow plot" for k=1,...,29.
    """
    # Read and sample the data, create the matrix (see read_data.py)
    data = load_sample(frac=0.1, random_state=1, save=False)

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