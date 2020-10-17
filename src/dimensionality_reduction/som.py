import SimpSOM as sps
import numpy as np
import os


class Som:
    def __init__(self, train_data, size=20, filepath="../results/som.npy"):
        self.train_data = train_data
        self.size = size
        self.net = sps.somNet(self.size, self.size, self.train_data, PBC=True)

        if not os.path.exists(filepath):
            self.train()
            self.save(filepath)
        else:
            self.load(filepath)

    def train(self, learning_rate=0.01, train_epochs=10000):
        self.net.train(learning_rate, train_epochs)

    def save(self, filename):
        self.net.save(filename)

    def load(self, filename):
        self.net = sps.somNet(self.size, self.size, self.train_data, PBC=True, loadFile=filename)

    def get_embeddings(self, data):
        return np.array(self.net.project(data, show=False, printout=False))


def som_embedd(data):
    som = Som(data)
    return som.get_embeddings(data)