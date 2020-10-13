import SimpSOM as sps


class SomClusterer:
    def __init__(self, train_data, size=20):
        self.train_data = train_data
        self.size = size
        self.net = sps.somNet(self.size, self.size, self.train_data, PBC=True)

    def train(self, learning_rate=0.01, train_epochs=10000):
        self.net.train(learning_rate, train_epochs)

    def get_clusters(self, data, plot=False):
        return self.net.cluster(data, show=plot, printout=False, savefile=False)

    def save(self, filename):
        self.net.save(filename)

    def load(self, filename):
        self.net = sps.somNet(self.size, self.size, self.train_data, PBC=True, loadFile=filename)


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
