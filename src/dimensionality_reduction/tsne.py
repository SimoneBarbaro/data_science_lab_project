from sklearn.manifold import TSNE

from dimensionality_reduction.embedder import Embedder


def tsne_dimred(data, **kwargs):
    embedded = TSNE(**kwargs).fit_transform(data)
    return embedded


class TsneEmbedder(Embedder):
    def __init__(self, **kwargs):
        self.tsne = TSNE(**kwargs)

    def embed(self, data):
        return self.tsne.fit_transform(data)