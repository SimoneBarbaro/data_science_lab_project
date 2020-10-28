class Embedder:
    """
    Interface of an embedder class.
    """
    def embed(self, data):
        """
        Method must return an embedding of the data, a two dimensional representation.
        :param data: the data to embed.
        :return: The data transformed into the two dimensional space.
        """
        raise NotImplementedError()
