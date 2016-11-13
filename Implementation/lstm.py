import numpy as np


class runLSTM:
    def __init__(self, featureSet):
        self.featureSet = featureSet


    def sigmoid(self, val):
        return 1/(1+np.exp(-val))

    def dsigmoid(self, val):
        return val*()


def main():
    dataSet = ld.importDataset()
    currSet = dataSet[:]

if __name__ == "__main__":
    main()



