def tsne_dimred(data, n_components=2, n_jobs=1):
    """
    Run TSNE to reduce dimensions to 2.
    :param data: Pandas dataframe of the data.
    :param n_components: Number of components for TSNE.
    :param n_jobs: Number of threads for parallelization (requires MulticoreTSNE on older versions of sklearn).
    :return: A numpy array of the TSNE embedding.
    """
    #if n_jobs == 1:
    from sklearn.manifold import TSNE
    #else:
    #    from MulticoreTSNE import MulticoreTSNE as TSNE  # pip install MulticoreTSNE
    embedded = TSNE(n_components=n_components, n_jobs=n_jobs).fit_transform(data)
    return embedded