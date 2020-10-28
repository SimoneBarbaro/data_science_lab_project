from umap import UMAP  # pip install umap-learn

from dimensionality_reduction.embedder import Embedder


def umap_dimred(data, **kwargs):
    embedded = UMAP(**kwargs).fit_transform(data)
    return embedded


class UmapEmbedder(Embedder):
    def __init__(self, **kwargs):
        self.umap = UMAP(**kwargs)

    def embed(self, data):
        return self.umap.fit_transform(data)
