def tsne_dimred(data, **kwargs):
    from sklearn.manifold import TSNE
    embedded = TSNE(**kwargs).fit_transform(data)
    return embedded