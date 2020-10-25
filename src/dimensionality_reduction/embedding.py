from src.dimensionality_reduction.som import Som
from src.dimensionality_reduction.tsne import TsneEmbedder
from src.dimensionality_reduction.umap import UmapEmbedder


def get_embedder(name, **kwargs):
    if name == "tsne":
        return TsneEmbedder(**kwargs)
    elif name == "umap":
        return UmapEmbedder(**kwargs)
    elif name == "som":
        return Som(**kwargs)
    else:
        raise NotImplementedError("embedder requested not implemented")
