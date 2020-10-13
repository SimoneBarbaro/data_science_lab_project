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
