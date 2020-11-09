from dimensionality_reduction.som import Som
from dimensionality_reduction.tsne import TsneEmbedder
from dimensionality_reduction.umap import UmapEmbedder


def get_embedder(name, **kwargs):
    """
    Return an embedder given the name.
    :param name: The embedder name.
    :param kwargs: Arguments to be passed to the embedder.
    :return: An embedder.
    """
    if name == "tsne":
        return TsneEmbedder(**kwargs)
    elif name == "umap":
        return UmapEmbedder(**kwargs)
    elif name == "som":
        return Som(**kwargs)
    else:
        raise NotImplementedError("embedder requested not implemented")
