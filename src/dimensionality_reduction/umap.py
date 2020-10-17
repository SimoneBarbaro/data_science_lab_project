def umap_dimred(data, **kwargs):
    from umap import UMAP  # pip install umap-learn
    embedded = UMAP(**kwargs).fit_transform(data)
    return embedded