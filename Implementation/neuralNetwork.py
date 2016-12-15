from sklearn.neural_network import MLPClassifier
import sys 

import svm



sys.path.append("./../Baseline")


import loadData as ld
import baseline
import numpy as np
import copy
import math
import random
from sklearn.model_selection import KFold, cross_val_score


from multiprocessing import Pool 
from multiprocessing import Process


class neuralNet(svm.runSVM):
	def __init__(self, dataset, classSet):
		self.dataset = dataset
		self.classSet = classSet

		self.splitDataset(10)


	def runClassification(self, train, test, foldNum, train_index, test_index):
		classificationTable = self.getActualClassification(train_index, False)

		clf = MLPClassifier(solver = 'adam', alpha = 1e-7, hidden_layer_sizes=(12,11), random_state = 1	)#, early_stopping = True)

		clf.fit(train, classificationTable)

		self.prediction = clf.predict(test)

		self.getActualClassification(test_index, True)









def main():
	cleanedSet, classSet = svm.loadAndClean()
	neuralNet(cleanedSet, classSet)

if __name__ == "__main__":
	main()
